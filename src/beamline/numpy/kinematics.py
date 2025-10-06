from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import hepunits as u
import numpy as np
import vector

from beamline.units import to_clhep, ureg

pytree = vector.register_pytree()

MUON_MASS: float = to_clhep(ureg.muon_mass)
MUON_CHARGE: float = to_clhep(ureg.elementary_charge)


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


pytree.register_node_class()(ParticleState)


def make_muon(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    t: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: int = -1,
) -> ParticleState:
    """Construct a muon with position in mm and momentum in MeV/c"""
    return ParticleState(
        position=vector.obj(t=t, x=x, y=y, z=z),
        momentum=vector.obj(
            px=px, py=py, pz=pz, m=MUON_MASS * u.c_light
        ).to_pxpypzenergy(),
        mass=MUON_MASS,
        charge=charge * MUON_CHARGE,
    )


@dataclass
class FieldStrength:
    r"""Electromagnetic field strength 2-form

    i.e. the antisymmetric tensor $F_{\mu\nu}$"""

    E: vector.VectorObject3D
    "Electric field in [MeV/e/mm]"
    B: vector.VectorObject3D
    "Magnetic field in [MeV*ns/e/mm^2]"

    def tree_flatten(self):
        return ((self.E, self.B), ())

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple,
        children: tuple[vector.VectorObject3D, vector.VectorObject3D],
    ):
        E, B = children
        return cls(E, B)

    def contract(self, p: vector.MomentumObject4D) -> vector.MomentumObject4D:
        r"""Contract the field 2-form with a momentum vector, and use the metric to raise the result to a vector

        Computes $\eta^{\rho\nu}F_{\mu\nu} p^{\nu}$
        where $\eta$ is the Minkowski metric with signature (+,-,-,-)

        `p` should be in units of MeV/c, i.e. (E/c, px, py, pz)
        The result is a 4-vector in units of [MeV^2 * ns^2 / e / mm^3]
        """
        # Pre-convert to cartesian for performance
        Etmp = (self.E / u.c_light).to_xyz()
        Btmp = self.B.to_xyz()
        return vector.MomentumObject4D(
            t=Etmp.x * p.x + Etmp.y * p.y + Etmp.z * p.z,
            px=Etmp.x * p.t + Btmp.z * p.y - Btmp.y * p.z,
            py=Etmp.y * p.t - Btmp.z * p.x + Btmp.x * p.z,
            pz=Etmp.z * p.t + Btmp.y * p.x - Btmp.x * p.y,
        )


pytree.register_node_class()(FieldStrength)


class EMTensorField(ABC):
    @abstractmethod
    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        """Evaluate the field tensor at a given position"""
        ...

    def __add__(self, other: Self):
        return SumField([self, other])


class SumField(EMTensorField):
    components: list[EMTensorField]

    def __init__(self, components: list[EMTensorField]):
        self.components = []
        for comp in components:
            if isinstance(comp, SumField):
                self.components.extend(comp.components)
            else:
                self.components.append(comp)

    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        E = vector.VectorObject3D(x=0.0, y=0.0, z=0.0)
        B = vector.VectorObject3D(x=0.0, y=0.0, z=0.0)
        for comp in self.components:
            F = comp.field_strength(position)
            E += F.E
            B += F.B
        return FieldStrength(E=E, B=B)


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
    # 1/c because force is dp/dtau
    dmomentum_dctau = (state.charge / state.mass / u.c_light) * field.field_strength(
        state.position
    ).contract(state.momentum)
    dtau_dt = state.momentum.gamma
    return ParticleState(
        position=dposition_dct.to_xyzt(),
        momentum=dmomentum_dctau / dtau_dt,
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
    dmomentum_dz = (state.charge / state.momentum.pz) * field.field_strength(
        state.position
    ).contract(state.momentum)
    return ParticleState(
        position=dposition_dz,
        momentum=dmomentum_dz,
        mass=0.0,
        charge=0.0,
    )
