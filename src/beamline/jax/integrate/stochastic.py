"""Stochastic muon propagation via operator splitting

This is a worked example of propagating a muon through electromagnetic fields
*and* material, where the material adds stochastic effects (energy straggling
now, multiple scattering later). As with ``diffrax_solve`` in ``propagate.py``,
you will probably want to write your own driver per use case; this one
demonstrates the intended structure.

Why not a diffrax SDE term?  Landau straggling is one-sided and heavy-tailed and
the Moliere scattering tail is Rutherford-like; neither is a subdivision-
consistent diffusion, so they do not fit diffrax's ``ControlTerm`` / Brownian-
path machinery. Putting ``jax.random`` inside the ODE right-hand side is also
wrong under adaptive Runge-Kutta, which evaluates the RHS at every stage and on
*rejected* steps. Instead we use **operator splitting**: the deterministic
Lorentz-force ODE is integrated with diffrax (keeping the RHS pure), and the
stochastic kick is applied as a discrete update between steps.

Stepping is done manually with ``solver.step`` (see
https://docs.kidger.site/diffrax/usage/manual-stepping/) so that we keep PID
adaptive step control where there is no material, while segmenting the traversal
by the material's characteristic length where there is. The
``BoundaryAwareStepSizeController`` is keyed on a signed distance combined over
the EM field and the material volume, so steps also stop cleanly at material
boundaries.

The driver is a nested loop: an outer ``lax.scan`` over the requested save
intervals (each lands exactly on the next grid point, so saving needs no
interpolation), and an inner ``eqx.internal.while_loop`` of at most
``max_substeps`` solver steps that integrates the interval and applies kicks.
The while loop terminates naturally when the interval endpoint is reached, so
surplus iterations are never run.

PRNG convention: the ``key`` is an explicit argument and is ``jr.split`` once per
step inside the loop. Ensemble runs ``vmap`` over a batch of keys.

Differentiability: there are two nested loops, each requiring its own AD
treatment.  The inner substep loop uses ``eqx.internal.while_loop``: the default
``kind="checkpointed"`` gives O(log n) memory for reverse-mode AD (``jax.grad``
/ ``jax.jacrev``) via recursive checkpointing; ``kind="lax"`` supports
forward-mode AD (``jax.jvp`` / ``jax.jacfwd``) efficiently but cannot be
reverse-differentiated.  The RK stage loop *inside* each ``solver.step`` call is
a separate level controlled by ``scan_kind`` on ``Dopri5``: ``None`` (the default
checkpointed ``custom_vjp``) pairs with ``kind="checkpointed"``, and ``"lax"``
pairs with ``kind="lax"``.  The ``forward_mode`` flag wires both together.
Reverse is the default since scalar-loss optimization has many parameters and one
output. Either way the reparameterized sampler makes the kicks differentiable in
their distribution parameters (mean energy loss, etc.), so gradients flow through
the physics, while the numerical step-size control is wrapped in
``stop_gradient`` (a discretization choice the converged solution is, to
tolerance, independent of).

TODO: a lot of the body of stochastic_solve is diffrax boilerplate, try to factorize
TODO: refactor apply_energy_loss to a general kick (scattering)
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import Dopri5, ODETerm, PIDController
from jax import Array, lax

from beamline.jax.absorber.material import StragglingParams
from beamline.jax.absorber.straggling import dummy_energy_loss_sampler
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.integrate.propagate import particle_interaction
from beamline.jax.integrate.stepsize import BoundaryAwareStepSizeController
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SFloat


def apply_energy_loss[T: ParticleState](state: T, dE: SFloat) -> T:
    """Reduce a particle's energy by ``dE`` [MeV], conserving direction

    The total energy is lowered by ``dE`` (floored at the rest mass), the
    momentum magnitude is recomputed from the on-shell relation, and the spatial
    momentum is rescaled to that magnitude (its direction unchanged). The
    position is untouched. This is a state update (cf. ``build_tangent``, which
    builds derivatives).
    """
    coords = state.kin.t.coords
    p3 = coords[..., :3]
    energy = coords[..., 3]
    mass = state.mass
    pmag = jnp.sqrt(jnp.sum(p3**2, axis=-1))
    energy_new = jnp.maximum(energy - dE, mass)
    # floor keeps the sqrt (and its derivative) finite under jax_debug_nans
    pmag_new = jnp.sqrt(jnp.maximum(energy_new**2 - mass**2, 0.0) + 1e-12)
    safe_pmag = jnp.where(pmag > 0.0, pmag, 1.0)
    scale = jnp.where(pmag > 0.0, pmag_new / safe_pmag, 0.0)
    new_coords = jnp.concatenate(
        [p3 * scale[..., None], energy_new[..., None]], axis=-1
    )
    return eqx.tree_at(lambda s: s.kin.t, state, Cartesian4(coords=new_coords))


def _combined_sdf(
    field: EMTensorField, material: MaterialVolume, state: ParticleState
) -> SFloat:
    """Signed distance to the nearest EM-field *or* material boundary

    Used only for step-size control, so the state is detached: the SDF's
    ``where(disc >= 0, sqrt, inf)`` would otherwise produce a NaN *tangent* under
    forward-mode autodiff (which evaluates tangents eagerly, before the
    ``stop_gradient`` on the controller's outputs would discard them).
    """
    ray = lax.stop_gradient(state).ray()
    return jax.lax.min(
        field.signed_distance(ray),
        material.signed_distance(ray),
    )


def stochastic_solve[T: ParticleState](
    field: EMTensorField,
    material: MaterialVolume,
    start: T,
    cts: Array,
    key: Array,
    *,
    sampler: Callable[
        [StragglingParams, Array], tuple[SFloat, SFloat]
    ] = dummy_energy_loss_sampler,
    forward_mode: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_substeps: int = 64,
) -> tuple[T, dict[str, Any]]:
    """Propagate a muon through ``field`` and ``material`` with stochastic kicks

    Args:
        field: The electromagnetic field to propagate through.
        material: The material volume that adds stochastic energy loss.
        start: The initial particle state (at ``cts[0]``).
        cts: Ascending grid of independent-variable values to save at; ``cts[0]``
            is the start and ``cts[-1]`` the end of integration. Consecutive
            points define the integration sub-intervals.
        key: A JAX PRNG key (``vmap`` a batch of keys for an ensemble).
        sampler: Energy-loss sampler ``(StragglingParams, key) -> (dE, log_w)``;
            pass ``landau_energy_loss_sampler`` (value/pathwise gradients) or
            ``landau_energy_loss_sampler_wg`` (weight/score-function gradients).
            The per-step ``log_w`` is accumulated and returned as
            ``stats["log_weight"]`` (0 for the value-gradient samplers); under an
            ensemble ``vmap`` the weighted estimator is ``sum(w f) / sum(w)`` with
            ``w = exp(log_weight)``.
        forward_mode: If True, configure the solver for forward-mode autodiff
            (``jax.jvp`` / ``jax.jacfwd``); otherwise (default) reverse-mode
            (``jax.grad`` / ``jax.jacrev``). See the module docstring.
        rtol, atol: Tolerances for the inner PID controller.
        max_substeps: Maximum number of solver steps per sub-interval (safety
            bound on the inner while loop). Must be large enough for the
            controller (plus material segmenting) to cross each interval.

    Returns:
        The saved states (batched along ``cts``) and a dict of solver stats.
    """
    term = ODETerm(particle_interaction)
    # Both loops (inner substep while_loop and inner RK stage loop) need to be
    # configured for the same AD direction; forward_mode wires them together.
    kind = "lax" if forward_mode else "checkpointed"
    solver = Dopri5(scan_kind="lax" if forward_mode else None)
    error_order = solver.error_order(term)
    initial_step, max_step = 1.0 * u.mm, 1.0 * u.m
    char_len = material.characteristic_length()

    controller = BoundaryAwareStepSizeController(
        PIDController(rtol=rtol, atol=atol, factormax=2.0),
        sdf=partial(_combined_sdf, field, material),
        max_step=max_step,
    )

    t0, t1 = cts[0], cts[-1]
    tnext0, controller_state = controller.init(
        term, t0, t1, start, initial_step, field, solver.func, error_order
    )
    # Step-size control is a discretization choice, not physics (see below).
    tnext0 = lax.stop_gradient(jnp.minimum(tnext0, t1))
    controller_state = lax.stop_gradient(controller_state)
    solver_state = solver.init(term, t0, tnext0, start, field)

    def in_material(state: ParticleState) -> Array:
        return material.contains(state.kin.p.to_cartesian3())

    def substep(carry, bound):
        """One manual solver step within a sub-interval ending at ``bound``"""
        (
            tprev,
            tnext,
            y,
            solver_state,
            controller_state,
            made_jump,
            key,
            log_weight,
            num_steps,
            num_accepted,
        ) = carry

        # Clamp the step to the interval end and (inside material) the
        # characteristic segment length.
        tnext_eff = jnp.minimum(tnext, bound)
        tnext_eff = jnp.where(
            in_material(y), jnp.minimum(tnext_eff, tprev + char_len), tnext_eff
        )

        y_cand, y_error, _dense, solver_state_cand, _ = solver.step(
            term, tprev, tnext_eff, y, field, solver_state, made_jump
        )
        y_error = jax.tree.map(lambda x: jnp.where(jnp.isnan(x), jnp.inf, x), y_error)

        keep_step, tprev_next, tnext_next, ctrl_jump, controller_state, _ = (
            controller.adapt_step_size(
                tprev,
                tnext_eff,
                y,
                y_cand,
                field,
                y_error,
                error_order,
                controller_state,
            )
        )
        # Step-size control is a discretization choice, not physics: don't let
        # gradients flow through the controller's timing decisions.
        tprev_next = lax.stop_gradient(jnp.minimum(tprev_next, bound))
        tnext_next = lax.stop_gradient(tnext_next)
        controller_state = lax.stop_gradient(controller_state)

        def keep(a, b):
            return jnp.where(keep_step, a, b)

        y_kept = jax.tree.map(keep, y_cand, y)
        solver_state = jax.tree.map(keep, solver_state_cand, solver_state)

        # Stochastic kick over the segment just traversed (accepted, in material).
        pos0 = y.kin.p.to_cartesian3().coords
        pos1 = y_cand.kin.p.to_cartesian3().coords
        # safe sqrt: zero-length (rejected) steps would give an infinite
        # derivative, so differentiate a floored value and zero it back out.
        sumsq = jnp.sum((pos1 - pos0) ** 2, axis=-1)
        thickness = jnp.where(
            sumsq > 0.0, jnp.sqrt(jnp.where(sumsq > 0.0, sumsq, 1.0)), 0.0
        )
        in_mat_step = material.contains(Cartesian3(coords=0.5 * (pos0 + pos1)))
        kick_applied = keep_step & in_mat_step

        params = material.interaction_params(y_kept, thickness)
        key, subkey = jr.split(key)
        dE_raw, logw_raw = sampler(params, subkey)
        dE = jnp.where(kick_applied, dE_raw, 0.0)
        # Accumulate the importance log-weight (outside any stop_gradient: the
        # weight-gradient estimator's gradient must flow through it).
        log_weight = log_weight + jnp.where(kick_applied, logw_raw, 0.0)
        y_new = apply_energy_loss(y_kept, dE)

        # A kick perturbs y, so the solver's cached (FSAL) derivative is stale:
        # signal a jump so it is recomputed next step.
        made_jump = jnp.where(keep_step, ctrl_jump | kick_applied, made_jump)

        tprev_out = jnp.where(keep_step, tprev_next, tprev)
        tnext_out = jnp.where(keep_step, tnext_next, tnext)
        return (
            tprev_out,
            tnext_out,
            y_new,
            solver_state,
            controller_state,
            made_jump,
            key,
            log_weight,
            num_steps + 1,
            num_accepted + jnp.where(keep_step, 1, 0),
        )

    def integrate_interval(carry, bound):
        """Integrate one sub-interval up to ``bound`` and emit the state there"""
        carry = eqxi.while_loop(
            lambda c: c[0] < bound,
            lambda c: substep(c, bound),
            carry,
            max_steps=max_substeps,
            kind=kind,
        )
        y = carry[2]
        return carry, (y.kin.p.coords, y.kin.t.coords)

    init_carry = (
        t0,
        tnext0,
        start,
        solver_state,
        controller_state,
        jnp.array(False),
        key,
        jnp.array(0.0),
        jnp.array(0),
        jnp.array(0),
    )
    final_carry, (saved_p, saved_t) = lax.scan(integrate_interval, init_carry, cts[1:])

    # Prepend the start state (at cts[0]) to the per-interval endpoints.
    save_p = jnp.concatenate([start.kin.p.coords[None], saved_p], axis=0)
    save_t = jnp.concatenate([start.kin.t.coords[None], saved_t], axis=0)
    ys = type(start)(
        kin=Tangent(p=Cartesian4(coords=save_p), t=Cartesian4(coords=save_t)),
        q=start.q,
    )
    stats = {
        "log_weight": final_carry[7],
        "num_steps": final_carry[8],
        "num_accepted_steps": final_carry[9],
        "num_rejected_steps": final_carry[8] - final_carry[9],
    }
    return ys, stats
