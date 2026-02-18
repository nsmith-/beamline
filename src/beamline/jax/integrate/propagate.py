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

from beamline.jax.emfield import EMTensorField, particle_interaction
from beamline.jax.kinematics import ParticleState


def propagate(_ct: Any, state: ParticleState, field: EMTensorField) -> ParticleState:
    """Propagate a particle state through an electromagnetic field for use with diffrax"""

    return particle_interaction(state, field)


def diffrax_solve[T: ParticleState](
    field: EMTensorField,
    start: T,
    cts: Array,
    forward_mode: bool = True,
) -> T:
    """An example solver for muon propagation through non-stochastic components using diffrax

    Probably you want to design your solver per your use case, this is just an example.

    Args:
        field: The electromagnetic field to propagate through
        start: The initial particle state
        cts: The positions along the beamline to solve at
        forward_mode: Whether to use forward-mode AD for the adjoint method
            (more efficient when there are more outputs than inputs)
    """
    sol = diffeqsolve(
        terms=ODETerm(propagate),
        solver=Dopri5(),
        t0=cts[0],
        t1=cts[-1],
        dt0=1.0 * u.mm,
        y0=start,
        args=field,
        saveat=SaveAt(ts=cts),
        adjoint=ForwardMode() if forward_mode else RecursiveCheckpointAdjoint(),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),
    )
    return sol.ys
