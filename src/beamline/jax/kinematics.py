from __future__ import annotations

from abc import abstractmethod
from typing import Self

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    Cylindric4,
    Point,
    Tangent,
)
from beamline.jax.types import SFloat, SInt
from beamline.units import MUON_CHARGE, MUON_MASS


class ParticleState(eqx.Module):
    kin: eqx.AbstractVar[Tangent[Cartesian4]]
    """Kinematic state of a particle in Cartesian coordinates

    The tangent vector is scaled by mass so it is the four-momentum.
    """
    q: eqx.AbstractVar[SInt]
    """Charge sign of the particle"""

    @property
    @abstractmethod
    def mass(self) -> SFloat:
        """Mass-energy of the particle [MeV]

        TODO: consider returning mass in MeV/c^2
        Can add rest_energy property if needed
        """

    @property
    @abstractmethod
    def charge(self) -> SFloat:
        """Charge of the particle"""

    @abstractmethod
    def build_tangent(self, dkin: Tangent[Cartesian4]) -> Self:
        """Return a the particle state structure with specified kinematics

        This is also an opportunity to specify any other flows.
        Any non-kinematic parts of the state (e.g. charge) should be static
        fields and copied over from the current instance.
        """

    def gamma(self) -> SFloat:
        """Compute the Lorentz factor gamma"""
        E = self.kin.dx.ct
        # return E / abs(self.kin.dx)
        return E / self.mass


class MuonState(ParticleState):
    """Abstract state"""

    @property
    def mass(self) -> SFloat:
        return MUON_MASS * u.c_light_sq

    @property
    def charge(self) -> SFloat:
        return self.q * MUON_CHARGE

    @classmethod
    def make(
        cls,
        *,
        position: Cartesian4 | Cylindric4,
        momentum: Cartesian3 | Cylindric3,
        q: SInt,
    ) -> Self:
        """Create a MuonState from position and momentum components"""
        pos = Point(x=position.to_cartesian())
        mom3 = momentum.to_cartesian()
        mom4 = Cartesian4.make(
            x=mom3.coords[0],
            y=mom3.coords[1],
            z=mom3.coords[2],
            ct=jnp.sqrt(mom3.coords.dot(mom3.coords) + MUON_MASS**2 * u.c_light**4),
        )
        tangent_vector = Tangent(point=pos, dx=mom4)
        return cls(kin=tangent_vector, q=q)


class MuonStateDct(MuonState):
    """Muon state, propagating with respect to coordinate time ct"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDct:
        return MuonStateDct(kin=dkin, q=self.q)


class MuonStateDz(MuonState):
    """Muon state, propagating with respect to longitudinal position z"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDz:
        dct_dz = self.kin.dx.ct / self.kin.dx.z
        dkin_dz = jax.tree.map(lambda x: x * dct_dz, dkin)
        return MuonStateDz(kin=dkin_dz, q=self.q)
