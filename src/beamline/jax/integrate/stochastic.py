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
"""

import functools
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Protocol, cast

import equinox as eqx
import equinox.internal as eqxi
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import Dopri5, ODETerm, PIDController
from jax import Array, lax

from beamline.jax.absorber.material import InteractionParams
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.emfield import EMTensorField
from beamline.jax.integrate.propagate import particle_interaction
from beamline.jax.integrate.stepsize import BoundaryAwareStepSizeController
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SFloat


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _probe_tangent(x: Array, tag: str) -> Array:
    """Pass-through identity; during ``jax.jvp`` prints the tangent of ``x``.

    Used only in debug mode to verify whether a quantity carries a gradient
    from upstream parameters.  The tangent printout is a side-effect of the
    JVP rule; the primal is unchanged so this is safe to insert anywhere.
    """
    return x


@_probe_tangent.defjvp
def _probe_tangent_jvp(tag: str, primals, tangents):
    (x,) = primals
    (tx,) = tangents
    jax.debug.print(f"  tangent[{tag}] = {{t}}", t=tx)
    return x, tx


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


def _perp_basis(n: Cartesian3) -> tuple[Cartesian3, Cartesian3]:
    """Orthonormal (u, v) spanning the plane perpendicular to unit vector n
    """
    near_z = jnp.abs(n.z) >= 0.9
    ref = Cartesian3.make(
        x=jnp.where(near_z, 1.0, 0.0), y=0.0, z=jnp.where(near_z, 0.0, 1.0)
    )
    u = ref.cross(n)
    u = u * (1.0 / abs(u))
    return u, n.cross(u)


def apply_scattering[T: ParticleState](
    state: T, theta_x: SFloat, theta_y: SFloat, y_x: SFloat, y_y: SFloat
) -> T:
    """The in-material lateral offset is applied in the same (u, v)
    basis as the angles, so the per-plane angle/offset correlation (rho =
    sqrt(3)/2) is preserved.
    """
    p3 = Cartesian3(coords=state.kin.t.coords[..., :3])
    pmag = abs(p3)
    n = p3 * (1.0 / jnp.where(pmag > 0.0, pmag, 1.0))
    u, v = _perp_basis(n)
    theta = jnp.sqrt(theta_x**2 + theta_y**2)
    axis_raw = v * theta_x - u * theta_y
    axis = Cartesian3(coords=jnp.where(theta > 0.0, axis_raw.coords, u.coords))
    rotation = Transform.make_axis_angle(axis, theta, Cartesian4.make())
    rotated = eqx.tree_at(lambda s: s.kin.t, state, rotation.to_global(state.kin.t))
    offset = u * y_x + v * y_y
    new_pos = Cartesian4.make(
        x=rotated.kin.p.x + offset.x,
        y=rotated.kin.p.y + offset.y,
        z=rotated.kin.p.z + offset.z,
        ct=rotated.kin.p.ct,
    )
    return eqx.tree_at(lambda s: s.kin.p, rotated, new_pos)

class StochasticKick(Protocol):
    """A stochastic interaction applied over one traversed segment.

    Composes sampling and application into one (state, params, key) -> state
    call. Kicks apply unconditionally; the solver gates whether the result
    takes effect (real in-material accepted steps only). Weight-carrying kicks
    accumulate into state.log_weight.
    """

    def __call__(
        self, state: ParticleState, params: InteractionParams, key: Array
    ) -> ParticleState: ...


def energy_loss_kick_factory(
    sampler: Callable[[InteractionParams, Array], tuple[SFloat, SFloat]],
) -> StochasticKick:
    """Build an energy-loss kick from an energy-loss sampler."""

    def kick[T: ParticleState](state: T, params: InteractionParams, key: Array) -> T:
        dE, logw = sampler(params, key)
        state = apply_energy_loss(state, dE)
        return eqx.tree_at(
            lambda s: s.log_weight, state, state.log_weight + logw
        )

    return kick


def scattering_kick_factory(
    sampler: Callable[
        [InteractionParams, Array], tuple[SFloat, SFloat, SFloat, SFloat]
    ],
) -> StochasticKick:
    """Build a multiple-scattering kick from a scattering sampler."""

    def kick[T: ParticleState](state: T, params: InteractionParams, key: Array) -> T:
        tx, ty, yx, yy = sampler(params, key)
        return apply_scattering(state, tx, ty, yx, yy)

    return kick


def _combined_sdf(
    field: EMTensorField, material: MaterialVolume, state: ParticleState
) -> SFloat:
    """Signed time to the nearest EM-field *or* material boundary

    Used only for step-size control, so all three arguments are detached: the
    SDF's ``where(disc >= 0, sqrt, inf)`` produces a NaN *tangent* under
    forward-mode autodiff (which evaluates tangents eagerly, before the
    ``stop_gradient`` on the controller's outputs would discard them).  This
    applies not only to the state but also to field/material parameters (e.g.
    a rotation angle phi inside a TransformMaterialVolume) that may carry
    forward-mode tangents when differentiating w.r.t. scene geometry.
    """
    ray = lax.stop_gradient(state).ray()
    return jax.lax.min(
        lax.stop_gradient(field).signed_time_to_boundary(ray),
        lax.stop_gradient(material).signed_time_to_boundary(ray),
    )


def stochastic_solve[T: ParticleState](
    field: EMTensorField,
    material: MaterialVolume,
    start: T,
    cts: Array,
    key: Array,
    *,
    kicks: Sequence[StochasticKick] = (),
    forward_mode: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_substeps: int = 64,
    debug: bool = False,
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
        kicks: Sequence of StochasticKick to apply each accepted in-material
            substep, in order. Each is ``(state, params, key) -> state`` and
            self-gates on ``params.thickness``. Build them with
            ``energy_loss_kick_factory`` / ``scattering_kick_factory``. Empty
            (the default) means only deterministic propagation.
        forward_mode: If True, configure the solver for forward-mode autodiff
            (``jax.jvp`` / ``jax.jacfwd``); otherwise (default) reverse-mode
            (``jax.grad`` / ``jax.jacrev``). See the module docstring.
        rtol, atol: Tolerances for the inner PID controller.
        max_substeps: Maximum number of solver steps per sub-interval (safety
            bound on the inner while loop). Must be large enough for the
            controller (plus material segmenting) to cross each interval.
        debug: If True, emit per-substep diagnostics via ``jax.debug.print``
            (JIT-compatible). Useful for verifying that kicks are applied and
            tracing where geometry gradients enter the computation.

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
            num_steps,
            num_accepted,
        ) = carry
        y = cast(ParticleState, y)

        # Clamp the step to the interval end and (inside material) the
        # characteristic segment length.
        tnext_eff = jnp.minimum(tnext, bound)
        tnext_eff = jnp.where(
            in_material(y), jnp.minimum(tnext_eff, tprev + char_len), tnext_eff
        )

        y_cand, y_error, _dense, solver_state_cand, _ = solver.step(
            term, tprev, tnext_eff, y, field, solver_state, made_jump
        )
        y_cand = cast(ParticleState, y_cand)
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
        ray0 = y.ray()
        ray1 = y_cand.ray()
        sdf0 = material.signed_time_to_boundary(ray0)
        sdf1 = material.signed_time_to_boundary(ray1)
        # entirely in material: use displacement
        # when crossing a boundary: use signed_time_to_boundary
        # TODO: in both cases, assumes path length is displacement in a straight line
        # though in the boundary crossing case abs(sdf0) + abs(sdf1) >= abs(pos1 - pos0)
        displacement = abs(ray1.p - ray0.p)
        thickness = jnp.where(
            (sdf0 < 0.0) & (sdf1 < 0.0),
            displacement,
            jnp.where(
                # pick whichever is inside
                sdf0 < 0.0,
                -sdf0,
                jnp.where(
                    sdf1 < 0.0,
                    displacement - sdf0,
                    0.0,  # not in material
                ),
            ),
        )
        kick_applied = keep_step & (thickness > 0.0)
        thickness = jnp.where(thickness == 0.0, 1.0, thickness)
        params = material.interaction_params(y_kept, thickness)

        y_new = y_kept
        for kick in kicks:
            key, subkey = jr.split(key)
            y_new = kick(y_new, params, subkey)

        # Gate the whole kick on a real, accepted, in-material step (kick_applied
        # already includes keep_step, so rejected steps revert here too). This
        # also reverts log_weight accumulation outside material.
        y_new = jax.tree.map(
            lambda a, b: jnp.where(kick_applied, a, b), y_new, y_kept
        )

        # A kick perturbs y, so the solver's cached (FSAL) derivative is stale:
        # signal a jump so it is recomputed next step.
        made_jump = jnp.where(keep_step, ctrl_jump | kick_applied, made_jump)

        tprev_out = jnp.where(keep_step, tprev_next, tprev)
        tnext_out = jnp.where(keep_step, tnext_next, tnext)

        if debug:
            # _probe_tangent prints the JVP tangent in forward-mode AD, to verify
            # whether thickness carries a gradient from upstream geometry params.
            thick_p = _probe_tangent(thickness, "thick")
            sdf_val = _combined_sdf(field, material, y_new)
            jax.debug.print(
                "substep: [{tprev}, {tnext_eff}] -> [{tprev_out}, {tnext_out}]"
                "  keep={keep} sdf0={sdf0} sdf1={sdf1} kicked={kicked}"
                "  thick={thick} sdf={sdf}",
                tprev=tprev, tnext_eff=tnext_eff,
                tprev_out=tprev_out, tnext_out=tnext_out,
                keep=keep_step, sdf0=sdf0, sdf1=sdf1,
                kicked=kick_applied, thick=thick_p, sdf=sdf_val,
            )

        return (
            tprev_out,
            tnext_out,
            y_new,
            solver_state,
            controller_state,
            made_jump,
            key,
            num_steps + 1,
            num_accepted + jnp.where(keep_step, 1, 0),
        )

    def integrate_interval(carry, bound):
        """Integrate one sub-interval up to ``bound`` and emit the state there"""
        if debug:
            jax.debug.print(
                "interval: tprev={tprev} -> bound={bound}", tprev=carry[0], bound=bound
            )
        carry = eqxi.while_loop(
            lambda c: c[0] < bound,
            lambda c: substep(c, bound),
            carry,
            max_steps=max_substeps,
            kind=kind,
        )
        y = carry[2]
        return carry, (y.kin.p.coords, y.kin.t.coords, y.log_weight)

    init_carry = (
        t0,
        tnext0,
        start,
        solver_state,
        controller_state,
        jnp.array(False),
        key,
        jnp.array(0),
        jnp.array(0),
    )
    final_carry, (saved_p, saved_t, saved_w) = lax.scan(
        integrate_interval, init_carry, cts[1:]
    )

    # Prepend the start state (at cts[0]) to the per-interval endpoints.
    save_p = jnp.concatenate([start.kin.p.coords[None], saved_p], axis=0)
    save_t = jnp.concatenate([start.kin.t.coords[None], saved_t], axis=0)
    save_w = jnp.concatenate([jnp.asarray(start.log_weight)[None], saved_w], axis=0)
    ys = type(start)(
        kin=Tangent(p=Cartesian4(coords=save_p), t=Cartesian4(coords=save_t)),
        q=start.q,
        log_weight=save_w,
    )
    final_state = final_carry[2]
    stats = {
        "log_weight": final_state.log_weight,
        "num_steps": final_carry[7],
        "num_accepted_steps": final_carry[8],
        "num_rejected_steps": final_carry[7] - final_carry[8],
    }
    return ys, stats
