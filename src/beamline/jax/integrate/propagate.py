from functools import partial
from typing import Any

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

from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.geometry import Volume
from beamline.jax.integrate.stepsize import BoundaryAwareStepSizeController
from beamline.jax.kinematics import ParticleState, TrackingStatus
from beamline.jax.types import SFloat, SInt


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
    # particles that are no longer tracked are frozen in place
    alive = state.is_alive()
    return jax.tree.map(lambda d: jnp.where(alive, d, jnp.zeros_like(d)), dstate)


def update_status[T: ParticleState](
    state: T,
    ct: SFloat,
    *,
    aperture: Volume | None = None,
    material: MaterialVolume | None = None,
) -> T:
    """Post-step processing: stop tracking a particle that is lost at ``ct``

    Checks, in order of precedence, whether an alive particle has left
    ``aperture`` (``HIT_APERTURE``), stopped in ``material``
    (``STOPPED_IN_MATERIAL``), or has kinematics invalid for its independent
    variable (``REVERSED``). If so, ``status`` and ``lost_at`` are set.
    Particles that were already lost keep their original status.
    """
    reason = jnp.where(
        state.kinematics_valid(), TrackingStatus.ALIVE, TrackingStatus.REVERSED
    )
    if material is not None:
        reason = jnp.where(
            material.is_stopped(state), TrackingStatus.STOPPED_IN_MATERIAL, reason
        )
    if aperture is not None:
        exited = ~aperture.contains(state.kin.p.to_cartesian3())
        reason = jnp.where(exited, TrackingStatus.HIT_APERTURE, reason)
    lost_now = state.is_alive() & (reason != TrackingStatus.ALIVE)
    status = jnp.where(lost_now, reason, state.status).astype(state.status.dtype)
    lost_at = jnp.where(lost_now, ct, state.lost_at)
    return state.with_status(status, lost_at)


def frozen_status_interaction[T: ParticleState](
    ct: Any, state: T, args: tuple[EMTensorField, tuple[SInt, SFloat]]
) -> T:
    """``particle_interaction`` for a state stripped of its status

    The solvers integrate ``state.without_status()``, and pass
    ``(field, (status, lost_at))`` as ``args``, since status only changes
    between steps. Lost particles are evaluated on a regularized state (see
    ``ParticleState.regularized``) so the masked-out derivative stays finite.
    """
    field, (status, lost_at) = args
    full = state.with_status(status, lost_at).regularized()
    return particle_interaction(ct, full, field).without_status()


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
        aperture: If given, the particle is lost when it leaves this volume.
            When a particle is lost (see ``update_status``), the solve
            terminates at the end of that step, ``status`` and ``lost_at`` are
            set, and the state at loss is repeated for all later save points.
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
    status = (start.status, start.lost_at)
    sol = diffeqsolve(
        terms=ODETerm(frozen_status_interaction),
        solver=Dopri5(),
        t0=cts[0],
        t1=cts[-1],
        dt0=initial_step,
        y0=start.without_status(),
        args=(field, status),
        saveat=SaveAt(ts=cts, t1=True),
        adjoint=ForwardMode() if forward_mode else RecursiveCheckpointAdjoint(),
        stepsize_controller=controller,
        event=Event(partial(_lost, aperture)),
    )
    ys = sol.ys.with_status(
        *(jnp.broadcast_to(x, sol.ts.shape + jnp.shape(x)) for x in status)
    )

    # On termination, diffrax saves the final state right after the last reached
    # save point, and fills the remainder with inf. Without termination, the final
    # state is saved last (at t1).
    ilast = jnp.argmax(jnp.where(jnp.isfinite(sol.ts), sol.ts, -jnp.inf))
    ct_end = sol.ts[ilast]
    end = update_status(jax.tree.map(lambda x: x[ilast], ys), ct_end, aperture=aperture)
    reached = cts <= ct_end

    def fill(saved: Array, final: Array) -> Array:
        mask = reached.reshape(-1, *([1] * (saved.ndim - 1)))
        return jnp.where(mask, saved[:-1], final)

    ys = jax.tree.map(fill, ys, end)
    stats = {**sol.stats, "lost": sol.result == RESULTS.event_occurred}
    return ys, stats


def _lost(
    aperture: Volume | None,
    t: Any,
    y: ParticleState,
    args: tuple[EMTensorField, tuple[SInt, SFloat]],
    **kwargs,
) -> Array:
    """Event condition: an alive particle is lost (see ``update_status``)"""
    _, status = args
    state = y.with_status(*status)
    return state.is_alive() & ~update_status(state, t, aperture=aperture).is_alive()
