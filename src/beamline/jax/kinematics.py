from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
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
from beamline.jax.types import SBool, SFloat, SInt
from beamline.units import MUON_CHARGE, MUON_MASS


class TrackingStatus(IntEnum):
    """Why a particle is no longer tracked (or ``ALIVE`` if it still is)

    Stored per particle as an integer array in ``ParticleState.status``.
    """

    ALIVE = 0
    HIT_APERTURE = 1
    """Left the tracking aperture (see ``geometry.BeamPipe``)"""
    STOPPED_IN_MATERIAL = 2
    """Momentum fell below the material's stopping threshold (see
    ``absorber.material.Material.stopping_momentum``)"""
    REVERSED = 3
    """Kinematics invalid for the independent variable, e.g. a ``MuonStateDz``
    whose longitudinal momentum is no longer (sufficiently) positive"""


MIN_PZ_FRACTION: float = 1e-3
"""``MuonStateDz`` is marked ``REVERSED`` once ``p_z <= MIN_PZ_FRACTION * |p|``"""


class ParticleState(eqx.Module):
    kin: eqx.AbstractVar[Tangent[Cartesian4]]
    """Kinematic state of a particle in Cartesian coordinates

    The tangent vector is scaled by mass so it is the four-momentum.
    """
    q: eqx.AbstractVar[SInt]
    """Charge sign of the particle"""
    log_weight: eqx.AbstractVar[SFloat]
    """Log importance weight (0 for an unweighted draw)"""
    status: eqx.AbstractVar[SInt]
    """Tracking status, a ``TrackingStatus`` value

    Particles that are not ``ALIVE`` are frozen in place (see
    ``propagate.particle_interaction`` and ``propagate.update_status``).
    Status is bookkeeping, not part of the integrated ODE state: the solvers
    strip it (with ``lost_at``) before integrating, see ``without_status``.
    """
    lost_at: eqx.AbstractVar[SFloat]
    """Independent variable value at which ``status`` left ``ALIVE``

    Meaningless (zero) while the particle is alive.
    """

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

    def is_alive(self) -> SBool:
        """Whether the particle is still being tracked"""
        return self.status == TrackingStatus.ALIVE

    def kinematics_valid(self) -> SBool:
        """Whether the kinematics are valid for this choice of independent variable

        A particle for which this is false is marked ``REVERSED``.
        """
        return jnp.ones(self.kin.t.coords.shape[:-1], dtype=bool)

    def without_status(self) -> Self:
        """Strip the (non-integrated) status bookkeeping: ``status`` and ``lost_at``

        The result is what the ODE solvers integrate. The integer ``status``
        would otherwise be promoted to float by the solver's state arithmetic.
        """
        return eqx.tree_at(lambda s: (s.status, s.lost_at), self, (None, None))

    def with_status(self, status: SInt, lost_at: SFloat) -> Self:
        """Inverse of ``without_status``"""
        return eqx.tree_at(
            lambda s: (s.status, s.lost_at),
            self,
            (status, lost_at),
            is_leaf=lambda x: x is None,
        )

    @abstractmethod
    def regularized(self) -> Self:
        """Replace the momentum of non-alive particles with a benign placeholder

        A particle that stopped in material has ``|p| ~ 0``, where the equations
        of motion (and material interaction parameters) are singular. Their
        contribution is masked out with ``jnp.where``, but the masked branch
        must still be finite or its gradient (and ``jax_debug_nans``) yields
        NaN. Evaluate on ``state.regularized()`` instead (the "double where"
        trick).

        Which placeholder is benign depends on the independent variable (it
        must keep ``scale()`` finite), so each subclass chooses one; see
        ``_with_lost_momentum``. Alive particles must be returned unchanged.
        """

    def _with_lost_momentum(self, p3: Cartesian3) -> Self:
        """Helper for ``regularized``: set non-alive particles' momentum to ``p3``

        The energy is set on shell.
        """
        energy = jnp.sqrt(abs(p3) ** 2 + self.mass**2)
        placeholder = jnp.concatenate([p3.coords, energy[..., None]], axis=-1)
        coords = jnp.where(self.is_alive()[..., None], self.kin.t.coords, placeholder)
        return eqx.tree_at(lambda s: s.kin.t, self, Cartesian4(coords=coords))

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

    def ray(self) -> Tangent[Cartesian3]:
        """Return the 3D ray (position, scaled tangent) used for signed distances

        The tangent will always be scaled by the independent variable of integration's
        rate of change (per ``self.scale()``)
        """
        return (
            Tangent(p=self.kin.p.to_cartesian3(), t=self.kin.t.to_cartesian3())
            * self.scale()
            / self.kin.t.ct
        )


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
        log_weight = jnp.zeros(p4.shape[:-1])
        status = jnp.full(p4.shape[:-1], TrackingStatus.ALIVE, dtype=jnp.int32)
        lost_at = jnp.zeros(p4.shape[:-1])
        return cls(
            kin=tangent_vector,
            q=q,
            log_weight=log_weight,
            status=status,
            lost_at=lost_at,
        )


class MuonStateDct(MuonState):
    """Muon state, propagating with respect to coordinate time ct"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""
    log_weight: SFloat
    status: SInt
    lost_at: SFloat

    def scale(self) -> SFloat:
        return 1.0

    def regularized(self) -> MuonStateDct:
        """Placeholder ``p = m`` along ``+z``

        ``scale()`` is constant, so only ``|p| > 0`` matters (for e.g.
        Highland's ``1 / (beta p)`` in the material interaction parameters).
        """
        return self._with_lost_momentum(Cartesian3.make(z=self.mass))

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDct:
        """Derivative of log_weight is zero; it changes only at kicks.

        status and lost_at are bookkeeping (see ``ParticleState.without_status``);
        their zero derivatives just give the result the same structure.
        """
        return MuonStateDct(
            kin=dkin,
            q=self.q,
            log_weight=jnp.zeros_like(self.log_weight),
            status=jnp.zeros_like(self.status),
            lost_at=jnp.zeros_like(self.lost_at),
        )


class MuonStateDz(MuonState):
    """Muon state, propagating with respect to longitudinal position z"""

    kin: Tangent[Cartesian4]
    """State of a muon particle"""
    q: SInt = eqx.field(static=True)
    """Sign of the muon charge (+1 or -1)"""
    log_weight: SFloat
    status: SInt
    lost_at: SFloat

    def scale(self) -> SFloat:
        # convert from d/dz to d/dct
        return self.kin.t.ct / self.kin.t.z

    def kinematics_valid(self) -> SBool:
        return self.kin.t.z > MIN_PZ_FRACTION * abs(self.kin.t.to_cartesian3())

    def regularized(self) -> MuonStateDz:
        """Placeholder ``p = m`` along ``+z``

        Besides ``|p| > 0``, ``scale() = E / p_z`` needs ``p_z`` bounded away
        from zero, so the placeholder points along the integration axis.
        """
        return self._with_lost_momentum(Cartesian3.make(z=self.mass))

    def build_tangent(self, dkin: Tangent[Cartesian4]) -> MuonStateDz:
        """Derivative of log_weight is zero; it changes only at kicks.

        status and lost_at are bookkeeping (see ``ParticleState.without_status``);
        their zero derivatives just give the result the same structure.
        """
        return MuonStateDz(
            kin=dkin,
            q=self.q,
            log_weight=jnp.zeros_like(self.log_weight),
            status=jnp.zeros_like(self.status),
            lost_at=jnp.zeros_like(self.lost_at),
        )
