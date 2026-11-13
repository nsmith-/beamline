"""Solenoid models"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from quadax import quadgk

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.elliptic import (
    elliprd_one_zero,
    elliprf_one_zero,
    elliprj,
)
from beamline.jax.emfield import EMTensorField
from beamline.jax.magnet.axial_expansion import (
    AXIS_SERIES_RADIUS,
    B_rhoexpansion,
    axisymmetric_magnet_field_strength,
    axisymmetric_magnet_potential,
)
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.types import SBool, SFloat
from beamline.units import MU0


def _hypot(x: SFloat, y: SFloat) -> SFloat:
    """sqrt(x**2 + y**2)

    Not jnp.hypot: its second and higher derivatives are wrong where |x| == |y|
    (e.g. d2/dx2 hypot(x, 1) is 0 instead of 1/2**1.5 at x = 1), which breaks
    the jet-based rho expansion and Hessians at e.g. |z +- L/2| == R.
    Overflow is not a concern at the scales used here.
    """
    return jnp.sqrt(x * x + y * y)


def _hypot_ratio(x: SFloat, y: SFloat) -> SFloat:
    """Calculate x/sqrt(x**2+y**2)"""
    return x / _hypot(x, y)


def _auxp12(k2: SFloat, gamma: SFloat) -> tuple[SFloat, SFloat]:
    """Auxiliary functions for implementing 10.1109/20.947050

    From Eqn. 4, 6, alternatively formulated using Carlson integrals
    for better numerical stability near rho = 0.

    Args:
        k2: Real value in (0, 1], k2 = 1 corresponds to rho = 0
        gamma: Real value in [-1, 1), gamma = -1 corresponds to rho = 0
    """
    zero = jnp.zeros_like(k2)
    one = jnp.ones_like(k2)
    Rf = elliprf_one_zero(k2, one)
    Rd = elliprd_one_zero(k2, one)
    Rj = elliprj(zero, k2, one, gamma**2)
    P1 = Rf - 2 / 3 * Rd
    P2 = Rf - gamma * (1 + gamma) / 3 * Rj
    return P1, P2


def _quadarg_rho(R: SFloat, rho: SFloat, z: SFloat) -> SFloat:
    """Argument for numerical quadrature of a wire loop"""
    Brho, _ = WireLoop(R=R, I=1.0).B(rho, z)
    return Brho


def _quadarg_z(R: SFloat, rho: SFloat, z: SFloat) -> SFloat:
    """Argument for numerical quadrature of a wire loop"""
    _, Bz = WireLoop(R=R, I=1.0).B(rho, z)
    return Bz


class ThinShellSolenoid(EMTensorField):
    R: SFloat
    """Shell radius [mm]"""
    jphi: SFloat
    """Surface current density [e/ns/mm]"""
    L: SFloat
    """Length of the solenoid [mm]"""

    def Bz_onaxis(self, z: SFloat):
        """Magnetic field on-axis (rho=0)"""
        halfL = self.L / 2
        left_term = _hypot_ratio(z + halfL, self.R)
        right_term = _hypot_ratio(z - halfL, self.R)
        Bz = MU0 * self.jphi / 2 * (left_term - right_term)
        return Bz

    def B_rhoexpansion(
        self, rho: SFloat, z: SFloat, order: int = 1
    ) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Uses a series expansion in rho around the on-axis value
        McDonald model, expanding in powers of rho around the on-axis field
        """
        return B_rhoexpansion(self.Bz_onaxis, rho, z, order)

    def _guard_shell(self, rho: SFloat) -> SFloat:
        """Avoid rho = R, where the elliptic integral iteration never converges"""
        threshold_R = 1e-5 * self.R  # smaller eps = longer elliptic integral loop
        return jax.lax.select(
            jnp.abs(rho - self.R) < threshold_R,
            self.R + jnp.sign(rho - self.R) * threshold_R,
            rho,
        )

    def A(self, rho: SFloat, z: SFloat) -> SFloat:
        """Vector potential A_phi at (rho, z)

        The textbook form (e.g. Callaghan:1960) is, with n = 4 R rho / (rho + R)^2,
        m = 4 R rho / (zeta^2 + (rho + R)^2) and zeta = z -+ L/2,

            A_phi = mu0 j R / pi [ zeta / sqrt(zeta^2 + (rho + R)^2)
                    * ((m + n - m n) / (m n) K(m) - E(m) / m + (n - 1) / n Pi(n, m)) ]

        evaluated between the ends. In Carlson form the K terms cancel exactly and
        the 1 / (m n) singularities with them, leaving

            A_phi = mu0 j R / (3 pi) [ beta (R_D(0, 1-m, 1) - gamma^2 R_J(0, 1-m, 1, gamma^2)) ]

        with beta and gamma as in :meth:`B_elliptic` (1 - n = gamma^2). On the
        axis the bracket vanishes like rho, so evaluate near the axis via
        :func:`axisymmetric_magnet_potential` instead.
        """
        rho = self._guard_shell(rho)
        rhopR, rhomR = rho + self.R, rho - self.R
        gamma2 = (rhomR / rhopR) ** 2

        def edge(xi: SFloat) -> SFloat:
            k2 = (xi**2 + rhomR**2) / (xi**2 + rhopR**2)
            one = jnp.ones_like(k2)
            beta = xi / _hypot(xi, rhopR)
            Rd = elliprd_one_zero(k2, one)
            Rj = elliprj(jnp.zeros_like(k2), k2, one, gamma2)
            return beta * (Rd - gamma2 * Rj)

        prefactor = MU0 * self.jphi * self.R / (3 * jnp.pi)
        return prefactor * (edge(z + self.L / 2) - edge(z - self.L / 2))

    def B_dA(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Compute the magnetic field by taking the curl of the vector potential"""
        A, dA_drho = jax.value_and_grad(self.A, argnums=0)(rho, z)
        dA_dz = jax.grad(self.A, argnums=1)(rho, z)
        Brho = -dA_dz
        Bz = dA_drho + A / rho
        return Brho, Bz

    def B_elliptic(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Using Caciagli Eqn. 3-6, massaged to avoid singularities, and using
        Carlson elliptic integrals for numerical stability.

        References:
            Callaghan:1960 https://ntrs.nasa.gov/citations/19980227402
            Conway https://doi.org/10.1109/20.947050
            Caciagli:2018 https://doi.org/10.1016/j.jmmm.2018.02.003
        """
        rho = self._guard_shell(rho)
        xip, xim = z + self.L / 2, z - self.L / 2
        rhopR, rhomR = rho + self.R, rho - self.R
        alphap, alpham = 1 / _hypot(xip, rhopR), 1 / _hypot(xim, rhopR)
        betap, betam = xip * alphap, xim * alpham
        gamma = rhomR / rhopR
        k2p, k2m = (
            (xip**2 + rhomR**2) / (xip**2 + rhopR**2),
            (xim**2 + rhomR**2) / (xim**2 + rhopR**2),
        )
        P1p, P2p = _auxp12(k2p, gamma)
        P1m, P2m = _auxp12(k2m, gamma)
        prefactor = MU0 * self.jphi * self.R / jnp.pi
        Brho = prefactor * (alphap * P1p - alpham * P1m)
        Bz = prefactor / rhopR * (betap * P2p - betam * P2m)
        return Brho, Bz

    def B_quadloop(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Compute the magnetic field using numerical quadrature on a wire loop model"""
        quadfun_rho = partial(_quadarg_rho, self.R, rho)
        quadfun_z = partial(_quadarg_z, self.R, rho)
        bounds = jnp.array([z - self.L / 2, z + self.L / 2])
        Brho, _ = quadgk(quadfun_rho, bounds)
        Bz, _ = quadgk(quadfun_z, bounds)
        return self.jphi * Brho, self.jphi * Bz

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # although there are surfaces, the field is in all space
        return jnp.inf

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        return axisymmetric_magnet_field_strength(
            self.B_elliptic, self.Bz_onaxis, point, AXIS_SERIES_RADIUS * self.R
        )

    def potential(self, point: Cartesian4) -> Tangent[Cartesian4]:
        return axisymmetric_magnet_potential(
            self.A, self.Bz_onaxis, point, AXIS_SERIES_RADIUS * self.R
        )


class ThickSolenoid(EMTensorField):
    Rin: SFloat
    """Shell inner radius [mm]"""
    Rout: SFloat
    """Shell outer radius [mm]"""
    jphi: SFloat
    """Volume current density [e/ns/mm^2]"""
    L: SFloat
    """Length of the solenoid [mm]"""

    def Bz_onaxis(self, z: SFloat) -> SFloat:
        """Magnetic field on-axis (rho=0), in closed form

        Integrates the thin shell on-axis field over the winding radius:
        Bz = mu0 j / 2 [zeta ln((Rout + hypot(Rout, zeta)) / (Rin + hypot(Rin, zeta)))]
        evaluated between zeta = z - L/2 and zeta = z + L/2.
        """

        def edge(zeta: SFloat) -> SFloat:
            return zeta * jnp.log(
                (self.Rout + _hypot(self.Rout, zeta))
                / (self.Rin + _hypot(self.Rin, zeta))
            )

        halfL = self.L / 2
        return MU0 * self.jphi / 2 * (edge(z + halfL) - edge(z - halfL))

    def B_shells(
        self, rho: SFloat, z: SFloat, num_shells: int = 8, vmap: bool = True
    ) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Integrates thin shells over the winding radius with Gauss-Legendre quadrature.
        Outside the winding (rho < Rin or rho > Rout) the integrand is smooth in R
        and the quadrature converges exponentially: at the 3.2 benchmark coil, 8
        shells reach ~1e-13 relative accuracy. Inside the winding the integrand has
        a kink at R = rho and convergence is only algebraic.

        Args:
            rho, z: Field point in the solenoid frame [mm]
            num_shells: Number of quadrature nodes (thin shells)
            vmap: Evaluate shells with vmap (default; faster) instead of scan
        """
        nodes, weights = np.polynomial.legendre.leggauss(num_shells)
        half_width = (self.Rout - self.Rin) / 2
        shell_radii = (self.Rout + self.Rin) / 2 + half_width * nodes
        shell_widths = half_width * weights

        def shell_contrib_R(R: SFloat, dR: SFloat) -> tuple[SFloat, SFloat]:
            thin_solenoid = ThinShellSolenoid(R=R, jphi=self.jphi * dR, L=self.L)
            return thin_solenoid.B_elliptic(rho, z)

        def shell_contrib_body(
            carry: tuple[SFloat, SFloat], shell: tuple[SFloat, SFloat]
        ) -> tuple[tuple[SFloat, SFloat], None]:
            Brho, Bz = shell_contrib_R(*shell)
            return (carry[0] + Brho, carry[1] + Bz), None

        if vmap:
            Brho, Bz = jax.vmap(shell_contrib_R)(shell_radii, shell_widths)
            return jnp.sum(Brho), jnp.sum(Bz)

        zero = jnp.zeros_like(rho)
        out, _ = jax.lax.scan(
            shell_contrib_body, (zero, zero), (shell_radii, shell_widths)
        )
        return out

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # although there are surfaces, the field is in all space
        return jnp.inf

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        return axisymmetric_magnet_field_strength(
            self.B_shells, self.Bz_onaxis, point, AXIS_SERIES_RADIUS * self.Rin
        )

    def A_shells(self, rho: SFloat, z: SFloat, num_shells: int = 8) -> SFloat:
        """Vector potential A_phi, integrating thin shells as in :meth:`B_shells`"""
        nodes, weights = np.polynomial.legendre.leggauss(num_shells)
        half_width = (self.Rout - self.Rin) / 2
        shell_radii = (self.Rout + self.Rin) / 2 + half_width * nodes
        shell_widths = half_width * weights

        def shell_contrib(R: SFloat, dR: SFloat) -> SFloat:
            return ThinShellSolenoid(R=R, jphi=self.jphi * dR, L=self.L).A(rho, z)

        return jnp.sum(jax.vmap(shell_contrib)(shell_radii, shell_widths))

    def potential(self, point: Cartesian4) -> Tangent[Cartesian4]:
        return axisymmetric_magnet_potential(
            self.A_shells, self.Bz_onaxis, point, AXIS_SERIES_RADIUS * self.Rin
        )
