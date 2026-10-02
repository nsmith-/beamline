from functools import partial
from typing import Any

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
from diffrax import (
    RESULTS,
    Dopri5,
    Event,
    ForwardMode,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    diffeqsolve,
)
from jax import Array

from beamline.jax.coordinates import Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.geometry import Volume
from beamline.jax.integrate.stepsize import BoundaryAwareStepSizeController
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SFloat


def particle_interaction[T: ParticleState](
    _ct: Any, state: T, field: EMTensorField
) -> T:
    """Compute the interaction of a particle with an electromagnetic field

    Returns a differential change in the particle state due to the Lorentz force,
    with respect to the independent variable (as specified by state.scale()).

    TODO: other independent variables (proper time, path length, etc.)
        (this could go here as a parameter, a new function, or be part of state.build_tangent)
    TODO: verlet integration / symplectic integrators ?
    """
    # Note: to have ctau be the independent variable, divide by mc^2 instead of E (kin.t.ct)
    # unitless in this convention
    dposition_dct = state.kin * (1 / state.kin.t.ct)
    # Unit: [MeV/mm]
    dmomentum_dct = (state.charge / state.kin.t.ct) * field(state.kin)
    # dkin is really type Tangent(Tangent(Cartesian4)) but we don't yet have
    # an abstraction to make the tangent bundle itself a manifold
    dkin = Tangent(
        p=dposition_dct.t * state.scale(),
        t=dmomentum_dct.t * state.scale(),
    )
    dstate = state.build_tangent(dkin)
    # lost particles are frozen in place
    alive = state.is_alive()
    return jax.tree.map(lambda d: jnp.where(alive, d, jnp.zeros_like(d)), dstate)


def apply_aperture[T: ParticleState](state: T, ct: SFloat, aperture: Volume) -> T:
    """Post-step processing: mark the particle lost at ``ct`` if outside ``aperture``

    Particles that were already lost keep their original ``lost_at``.
    """
    exited = state.is_alive() & ~aperture.contains(state.kin.p.to_cartesian3())
    return eqx.tree_at(lambda s: s.lost_at, state, jnp.where(exited, ct, state.lost_at))


def sdf(
    field: EMTensorField, state: ParticleState, aperture: Volume | None = None
) -> SFloat:
    """Signed time to the nearest EM-field (or aperture) boundary, used for boundary-aware step size control

    Free function instead of lambda so BoundaryAwareStepSizeController can do its type inference.
    """
    ray = state.ray()
    dist = field.signed_time_to_boundary(ray)
    if aperture is not None:
        dist_aperture = aperture.signed_time_to_boundary(ray)
        dist = jnp.where(jnp.abs(dist_aperture) < jnp.abs(dist), dist_aperture, dist)
    return dist


def diffrax_solve[T: ParticleState](
    field: EMTensorField,
    start: T,
    cts: Array,
    *,
    aperture: Volume | None = None,
    forward_mode: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    debug: bool = False,
) -> tuple[T, dict[str, Any]]:
    """An example solver for muon propagation through non-stochastic components using diffrax

    Probably you want to design your solver per your use case, this is just an example.

    Args:
        field: The electromagnetic field to propagate through
        start: The initial particle state
        cts: The positions along the beamline to solve at
        aperture: If given, the particle is lost when it leaves this volume: the
            solve terminates at the end of that step, ``lost_at`` is set, and the
            state at loss is repeated for all later save points.
        forward_mode: Whether to use forward-mode AD for the adjoint method
            (more efficient when there are more outputs than inputs)

    Returns:
        A tuple of the solution at the specified positions, and the solver statistics
    """
    # TODO: depends on the independent variable, default being ct
    # initial_step should be 1/2 the minimum feature size
    initial_step, max_step = 1.0 * u.mm, 1.0 * u.m
    controller = BoundaryAwareStepSizeController(
        PIDController(rtol=rtol, atol=atol, factormax=2.0),
        sdf=partial(sdf, field, aperture=aperture),
        max_step=max_step,
        debug=debug,
    )
    # lost_at is inf while alive, which diffrax's dense interpolation would turn
    # into NaN, so it is held fixed in args rather than integrated
    sol = diffeqsolve(
        terms=ODETerm(_interaction_fixed_lost_at),
        solver=Dopri5(),
        t0=cts[0],
        t1=cts[-1],
        dt0=initial_step,
        y0=_strip_lost_at(start),
        args=(field, start.lost_at),
        saveat=SaveAt(ts=cts, t1=aperture is not None),
        adjoint=ForwardMode() if forward_mode else RecursiveCheckpointAdjoint(),
        stepsize_controller=controller,
        event=None if aperture is None else Event(partial(_exited, aperture)),
    )
    lost_at = jnp.broadcast_to(start.lost_at, sol.ts.shape)
    ys = _restore_lost_at(sol.ys, lost_at)
    if aperture is None:
        return ys, sol.stats

    # On termination, diffrax saves the final state right after the last reached
    # save point, and fills the remainder with inf. Without termination, the final
    # state is saved last (at t1).
    ilast = jnp.argmax(jnp.where(jnp.isfinite(sol.ts), sol.ts, -jnp.inf))
    ct_end = sol.ts[ilast]
    end = apply_aperture(jax.tree.map(lambda x: x[ilast], ys), ct_end, aperture)
    reached = cts <= ct_end

    def fill(saved: Array, final: Array) -> Array:
        mask = reached.reshape(-1, *([1] * (saved.ndim - 1)))
        return jnp.where(mask, saved[:-1], final)

    ys = jax.tree.map(fill, ys, end)
    stats = {**sol.stats, "lost": sol.result == RESULTS.event_occurred}
    return ys, stats


def _strip_lost_at[T: ParticleState](state: T) -> T:
    return eqx.tree_at(lambda s: s.lost_at, state, None)


def _restore_lost_at[T: ParticleState](state: T, lost_at: Array) -> T:
    return eqx.tree_at(lambda s: s.lost_at, state, lost_at, is_leaf=lambda x: x is None)


def _interaction_fixed_lost_at[T: ParticleState](
    ct: Any, state: T, args: tuple[EMTensorField, SFloat]
) -> T:
    """particle_interaction for a state stripped of lost_at (see diffrax_solve)"""
    field, lost_at = args
    dstate = particle_interaction(ct, _restore_lost_at(state, lost_at), field)
    return _strip_lost_at(dstate)


def _exited(
    aperture: Volume,
    t: Any,
    y: ParticleState,
    args: tuple[EMTensorField, SFloat],
    **kwargs,
) -> Array:
    """Event condition: an alive particle has left the aperture"""
    _, lost_at = args
    alive = jnp.isinf(lost_at)
    return alive & ~aperture.contains(y.kin.p.to_cartesian3())
