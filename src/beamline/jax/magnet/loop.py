"""Simple wire loop model"""

import jax.numpy as jnp

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Tangent,
)
from beamline.jax.elliptic import elliprd_one_zero, elliprf_one_zero
from beamline.jax.emfield import EMTensorField
from beamline.jax.magnet.axial_expansion import (
    AXIS_SERIES_RADIUS,
    B_rhoexpansion,
    axisymmetric_field_strength,
)
from beamline.jax.types import SBool, SFloat
from beamline.units import MU0

WIRE_GUARD = 1e-8
"""Minimum distance to the wire, relative to the loop radius, at which B is evaluated"""


class WireLoop(EMTensorField):
    R: SFloat
    """Radius of the wire loop [mm]"""
    I: SFloat  # noqa: E741
    """Current through the wire loop [e/ns]"""

    def Bz_onaxis(self, z: SFloat) -> SFloat:
        """Magnetic field on-axis (rho=0)"""
        return MU0 * self.I * self.R**2 / (2 * (self.R**2 + z**2) ** 1.5)

    def B_rhoexpansion(
        self, rho: SFloat, z: SFloat, order: int = 1
    ) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Uses a series expansion in rho around the on-axis value
        """
        return B_rhoexpansion(self.Bz_onaxis, rho, z, order)

    def B(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Magnetic field of a wire loop at (rho, z)

        The usual form in terms of K(m) and E(m), with m = 4 R rho / beta^2,
        has a 0/0 in Brho at rho = 0. Substituting the Carlson forms
        K = R_F(0, 1-m, 1) and E = R_F - m/3 R_D(0, 1-m, 1), the 1/rho cancels
        analytically, leaving expressions that are smooth through the axis
        (where 1-m = 1, R_F = pi/2, R_D = 3pi/4 and Brho -> 0).

        Not valid close to the wire: the thin-wire field diverges there, while a
        real conductor has finite size, and a particle that close would already
        have hit the beam pipe or the magnet casing. Points within
        ``WIRE_GUARD * R`` of the wire are only moved out to that distance so
        the R_F/R_D iteration terminates (exactly on the wire 1-m = 0 and it
        never would).

        Args:
            rho: Radial distance from the axis of the loop [mm]
            z: Axial distance from the plane of the loop [mm]

        Returns:
            (Brho, Bz): Tuple containing the radial and axial components of the magnetic field

        References:
            Granum:2022dtk https://doi.org/10.1016/j.nima.2022.166706
            https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html
        """
        rho, z = self._guard_wire(rho, z)
        R = self.R
        s = R**2 + rho**2 + z**2
        # computed directly (not as s -+ 2 R rho) to avoid cancellation near the wire
        alpha2 = (R - rho) ** 2 + z**2  # squared distance to the wire
        beta2 = (R + rho) ** 2 + z**2
        kp2 = alpha2 / beta2  # complementary parameter 1 - m
        one = jnp.ones_like(kp2)
        Rf = elliprf_one_zero(kp2, one)
        Rd = elliprd_one_zero(kp2, one)
        prefactor = MU0 * self.I * R / (jnp.pi * jnp.sqrt(beta2) * alpha2)
        Brho = prefactor * z * (Rf - 2 * s / (3 * beta2) * Rd)
        Bz = prefactor * (
            (R - rho) * Rf - 2 * rho * (R**2 - rho**2 - z**2) / (3 * beta2) * Rd
        )
        return Brho, Bz

    def _guard_wire(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Push (rho, z) out to at least WIRE_GUARD * R from the wire"""
        dmin = WIRE_GUARD * self.R
        drho = rho - self.R
        d = jnp.hypot(drho, z)
        near = d < dmin
        # exactly on the wire: pick the outward radial direction
        dsafe = jnp.where(d > 0, d, 1.0)
        urho = jnp.where(d > 0, drho / dsafe, 1.0)
        uz = jnp.where(d > 0, z / dsafe, 0.0)
        return (
            jnp.where(near, self.R + dmin * urho, rho),
            jnp.where(near, dmin * uz, z),
        )

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        return jnp.inf

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        return axisymmetric_field_strength(
            self.B, self.Bz_onaxis, point, AXIS_SERIES_RADIUS * self.R
        )
