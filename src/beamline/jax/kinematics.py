from __future__ import annotations

from abc import abstractmethod
from typing import Self

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    Cylindric4,
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
    def scale(self) -> SFloat:
        """Scaling factor for the tangent vector, used for integration

        This effectively determines the independent variable for integration.
        The default scaling is by lab time, but this allows alternative
        independent variables (e.g. proper time, path length, etc.)
        """

    @abstractmethod
    def build_tangent(self, dkin: Tangent[Cartesian4]) -> Self:
        """Build particle state structure with specified tangent vector

        Any non-kinematic parts of the state (e.g. charge) should be static
        fields and copied over from the current instance.
        """

    def beta(self) -> SFloat:
        """Compute the velocity beta = v/c"""
        E = self.kin.t.ct
        p = jnp.sum(self.kin.t.coords[..., :3] ** 2, axis=-1) ** 0.5
        return p / E

    def gamma(self) -> SFloat:
        """Compute the Lorentz factor gamma"""
        E = self.kin.t.ct
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
        pos = position.to_cartesian()
        mom3 = momentum.to_cartesian()
        mom4 = Cartesian4.make(
            x=mom3.coords[..., 0],
            y=mom3.coords[..., 1],
            z=mom3.coords[..., 2],
            ct=jnp.sqrt(mom3.coords.dot(mom3.coords) + MUON_MASS**2 * u.c_light**4),
        )
        p4, t4 = jnp.broadcast_arrays(pos.coords, mom4.coords)
        tangent_vector = Tangent(p=Cartesian4(p4), t=Cartesian4(t4))
        return cls(kin=tangent_vector, q=q)


class MuonStateDct(MuonState):
    """Muon state, propagating with respect to coordinate time ct"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""

    def scale(self) -> SFloat:
        return 1.0

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDct:
        return MuonStateDct(kin=dkin, q=self.q)


class MuonStateDz(MuonState):
    """Muon state, propagating with respect to longitudinal position z"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""

    def scale(self) -> SFloat:
        # convert from d/dz to d/dct
        return self.kin.t.ct / self.kin.t.z

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDz:
        return MuonStateDz(kin=dkin, q=self.q)
