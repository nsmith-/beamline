from functools import partial
from typing import Any

import hepunits as u
from diffrax import (
    Dopri5,
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
    return state.build_tangent(dkin)


def sdf(field: EMTensorField, state: ParticleState) -> SFloat:
    """A signed distance function for the electromagnetic field, used for boundary-aware step size control"""
    kin3 = (
        Tangent(
            p=state.kin.p.to_cartesian3(),
            t=state.kin.t.to_cartesian3(),
        )
        * state.scale()
        / state.kin.t.ct
    )
    return field.signed_distance(kin3)


def diffrax_solve[T: ParticleState](
    field: EMTensorField,
    start: T,
    cts: Array,
    *,
    forward_mode: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> tuple[T, dict[str, Any]]:
    """An example solver for muon propagation through non-stochastic components using diffrax

    Probably you want to design your solver per your use case, this is just an example.

    Args:
        field: The electromagnetic field to propagate through
        start: The initial particle state
        cts: The positions along the beamline to solve at
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
        sdf=partial(sdf, field),
        max_step=max_step,
    )
    sol = diffeqsolve(
        terms=ODETerm(particle_interaction),
        solver=Dopri5(),
        t0=cts[0],
        t1=cts[-1],
        dt0=initial_step,
        y0=start,
        args=field,
        saveat=SaveAt(ts=cts),
        adjoint=ForwardMode() if forward_mode else RecursiveCheckpointAdjoint(),
        stepsize_controller=controller,
    )
    return sol.ys, sol.stats
