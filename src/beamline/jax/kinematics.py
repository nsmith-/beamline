from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    Cylindric4,
    Point,
    TangentVector,
)
from beamline.jax.types import SFloat, SInt
from beamline.units import MUON_CHARGE, MUON_MASS


class ParticleState(eqx.Module):
    kin: eqx.AbstractVar[TangentVector[Cartesian4]]
    """Kinematic state of a particle in Cartesian coordinates

    The tangent vector is scaled by mass so it is the four-momentum.
    """

    @property
    @abstractmethod
    def mass(self) -> SFloat:
        """Mass-energy of the particle [MeV]"""

    @property
    @abstractmethod
    def charge(self) -> SFloat:
        """Charge of the particle"""

    @abstractmethod
    def with_kinematics(self, kin: TangentVector[Cartesian4]) -> ParticleState:
        """Return a the particle state structure with specified kinematics

        This is also an opportunity to specify any other flows. Anything that is not
        to be changed should be set to 0.
        """

    def gamma(self) -> SFloat:
        """Compute the Lorentz factor gamma"""
        E = self.kin.dx.coords[3]
        # return E / abs(self.kin.dx)
        return E / self.mass


class MuonState(ParticleState):
    kin: TangentVector[Cartesian4]
    """State of a muon particle"""
    q: SInt
    """Sign of the muon charge (+1 or -1)"""

    @property
    def mass(self) -> SFloat:
        return MUON_MASS * u.c_light_sq

    @property
    def charge(self) -> SFloat:
        return self.q * MUON_CHARGE

    def with_kinematics(self, kin: TangentVector[Cartesian4]) -> MuonState:
        return MuonState(kin=kin, q=0)

    @classmethod
    def make(
        cls,
        *,
        position: Cartesian4 | Cylindric4,
        momentum: Cartesian3 | Cylindric3,
        q: SInt,
    ) -> MuonState:
        """Create a MuonState from position and momentum components"""
        pos = Point(x=position.to_cartesian())
        mom3 = momentum.to_cartesian()
        mom4 = Cartesian4.make(
            x=mom3.coords[0],
            y=mom3.coords[1],
            z=mom3.coords[2],
            ct=jnp.sqrt(mom3.coords.dot(mom3.coords) + MUON_MASS**2 * u.c_light**4),
        )
        tangent_vector = TangentVector(point=pos, dx=mom4)
        return cls(kin=tangent_vector, q=q)
