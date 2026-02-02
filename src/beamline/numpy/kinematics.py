from dataclasses import dataclass

import hepunits as u
import numpy as np
import vector

from beamline.numpy.emfield import EMTensorField
from beamline.units import MUON_CHARGE, MUON_MASS

pytree = vector.register_pytree()


def polar_tangents(
    vec: vector.VectorObject,
) -> tuple[vector.VectorObject2D, vector.VectorObject2D]:
    """Given a 4-vector, return the unit vectors in the transverse plane corresponding to the polar coordinates

    Returns (rhohat, phihat) where rhohat points in the direction of increasing transverse radius
    and phihat points in the direction of increasing azimuthal angle phi.

    The returned vectors are 2D.
    """
    polar = vec.to_2D()
    return (
        vector.obj(rho=1.0, phi=polar.phi),
        vector.obj(rho=1.0, phi=polar.phi + np.pi / 2),
    )


@dataclass
class ParticleState:
    position: vector.VectorObject4D
    "Position vector in [mm] (including time dimension, i.e. ct)"
    momentum: vector.MomentumObject4D
    "Momentum vector in [MeV/c] (including energy dimension, i.e. E/c)"
    mass: float
    "Mass in [MeV/c^2]"
    charge: float
    "Charge in units of elementary charge"

    def __post_init__(self):
        # Ensure cartesian
        self.position = self.position.to_xyzt()
        self.momentum = self.momentum.to_pxpypzenergy()
        if self.mass < 0.0:
            raise ValueError("Mass must be positive")
        elif self.mass == 0.0:
            # This is assumped to be a tangent vector, so no checks
            return
        # expected = self.mass * u.c_light
        # if abs(self.momentum.mass - expected) / expected > 1e-2:
        #     msg = f"Momentum mass does not match state mass: {self.momentum.m} != {expected}"
        #     raise ValueError(msg)

    def tree_flatten(self):
        return ((self.position, self.momentum), (self.mass, self.charge))

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple[float, float],
        children: tuple[vector.VectorObject4D, vector.MomentumObject4D],
    ):
        pos, mom = children
        mass, charge = metadata
        return cls(pos, mom, mass, charge)


pytree.register_node_class()(ParticleState)  # type: ignore[arg-type]


def make_muon(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    ct: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: int = -1,
) -> ParticleState:
    """Construct a muon with position in mm and momentum in MeV/c"""
    return ParticleState(
        position=vector.obj(t=ct, x=x, y=y, z=z),
        momentum=vector.obj(
            px=px, py=py, pz=pz, m=MUON_MASS * u.c_light
        ).to_pxpypzenergy(),
        mass=MUON_MASS,
        charge=charge * MUON_CHARGE,
    )


def ode_tangent_dct(
    field: EMTensorField, _ct: float, state: ParticleState
) -> ParticleState:
    """Compute the tangent vector dstate/d(ct) for a particle in an EM field

    This propagates the state with respect to the time coordinate. To propagate
    with respect to the z coordinate, use `ode_tangent_dz`.
    Typically this function is partially applied with the field tensor and
    then passed to an ODE solver such as `beamline.numpy.integrate.solve_ivp`.
    """
    # We should have _ct == state.position.t to within the precision of the integrator
    dposition_dct = state.momentum / state.momentum.E
    Fuv = field.field_strength(state.position)
    # 1/c because force is dp/dtau and we want dp/d(ctau)
    dmomentum_dctau = (state.charge / state.mass / u.c_light) * Fuv.contract(
        state.momentum
    )
    # Note: we could also use dtau_dt = 1 / gamma = state.mass / state.momentum.energy
    # but in testing, this appears to have worse numerical stability
    dt_dtau = state.momentum.gamma
    return ParticleState(
        position=dposition_dct.to_xyzt(),
        momentum=dmomentum_dctau / dt_dtau,
        mass=0.0,
        charge=0.0,
    )


def ode_tangent_dz(
    field: EMTensorField, _z: float, state: ParticleState
) -> ParticleState:
    """Compute the tangent vector dstate/dz for a particle in an EM field

    This propagates the state with respect to the z coordinate. To propagate
    with respect to the time coordinate, use `ode_tangent_dct`.
    Typically this function is partially applied with the field tensor and
    then passed to an ODE solver such as `beamline.numpy.integrate.solve_ivp`.
    """
    # We should have _z == state.position.z to within the precision of the integrator
    dposition_dz = state.momentum / state.momentum.pz
    Fuv = field.field_strength(state.position)
    dmomentum_dz = (state.charge / state.momentum.pz) * Fuv.contract(state.momentum)
    return ParticleState(
        position=dposition_dz,
        momentum=dmomentum_dz,
        mass=0.0,
        charge=0.0,
    )
