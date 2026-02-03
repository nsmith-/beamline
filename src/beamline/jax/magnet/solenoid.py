"""Solenoid models"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.scipy.special import gamma
from quadax import quadgk

from beamline.jax.coordinates import Cartesian3, Cartesian4, Cylindric3, Point, Tangent
from beamline.jax.elliptic import (
    elliprd_one_zero,
    elliprf_one_zero,
    elliprj,
    elliptic_kepi,
)
from beamline.jax.emfield import EMTensorField
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.types import SFloat
from beamline.units import MU0


def _hypot_ratio(x: SFloat, y: SFloat) -> SFloat:
    """Calculate x/sqrt(x**2+y**2)"""
    hypot = jnp.hypot(x, y)
    return x / hypot


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
        Bz0, dnBz = jet(
            self.Bz_onaxis,
            primals=(z,),
            series=(jnp.zeros(2 * order + 1).at[0].set(1.0),),
        )
        hrho = rho / 2
        Bz = Bz0 + sum(
            (-1) ** n / gamma(n + 1) ** 2 * hrho ** (2 * n) * dnBz[2 * n - 1]
            for n in range(1, order + 1)
        )
        Brho = sum(
            (-1) ** (n + 1)
            / (n + 1)
            / gamma(n + 1) ** 2
            * hrho ** (2 * n + 1)
            * dnBz[2 * n]
            for n in range(order + 1)
        )
        return Brho, Bz

    def A(self, rho: SFloat, z: SFloat) -> SFloat:
        """Vector potential A_phi at (rho, z)

        Following Wikipedia"""
        raise NotImplementedError("Not yet tested")
        xip, xim = z + self.L / 2, z - self.L / 2
        rhopR = rho + self.R
        fourRrho = 4 * self.R * rho
        n = fourRrho / rhopR**2
        mp, mm = (
            fourRrho / (xip**2 + rhopR**2),
            fourRrho / (xim**2 + rhopR**2),
        )
        ap, am = xip / jnp.hypot(xip, rhopR), xim / jnp.hypot(xim, rhopR)
        Kp, Ep, Pip = elliptic_kepi(n=n, k=jnp.sqrt(mp))
        Km, Em, Pim = elliptic_kepi(n=n, k=jnp.sqrt(mm))
        termp = ap * ((mp + n - mp * n) / (mp * n) * Kp - Ep / mp + (n + 1) / n * Pip)
        termm = am * ((mm + n - mm * n) / (mm * n) * Km - Em / mm + (n + 1) / n * Pim)
        Aphi = MU0 * self.jphi * self.R / jnp.pi / self.L * (termp - termm)
        return Aphi

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
        # avoid rho = R which never converges
        threshold_R = 1e-5 * self.R  # smaller eps = longer elliptic integral loop
        rho = jax.lax.select(
            jnp.abs(rho - self.R) < threshold_R,
            self.R + jnp.sign(rho - self.R) * threshold_R,
            rho,
        )
        xip, xim = z + self.L / 2, z - self.L / 2
        rhopR, rhomR = rho + self.R, rho - self.R
        alphap, alpham = 1 / jnp.hypot(xip, rhopR), 1 / jnp.hypot(xim, rhopR)
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

    def field_strength(
        self, point: Point[Cartesian4]
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        xcyl = point.x.to_cylindric3()
        Brho, Bz = self.B(xcyl.rho, xcyl.z)
        Bphi = jnp.zeros_like(Brho)
        E = Tangent(Point(x=point.x.to_cartesian3()), dx=Cartesian3.make())
        B = Tangent(Point(x=xcyl), dx=Cylindric3.make(rho=Brho, phi=Bphi, z=Bz))
        return E, B.to_cartesian()


class ThickSolenoid(EMTensorField):
    Rin: SFloat
    """Shell inner radius [mm]"""
    Rout: SFloat
    """Shell outer radius [mm]"""
    jphi: SFloat
    """Volume current density [e/ns/mm^2]"""
    L: SFloat
    """Length of the solenoid [mm]"""

    def B_shells(
        self, rho: SFloat, z: SFloat, num_shells: int = 200
    ) -> tuple[SFloat, SFloat]:
        dR = (self.Rout - self.Rin) / num_shells

        # TODO: investigate scan vs. vmap
        def shell_contrib(
            carry: tuple[SFloat, SFloat], R: SFloat
        ) -> tuple[tuple[SFloat, SFloat], None]:
            thin_solenoid = ThinShellSolenoid(R=R, jphi=self.jphi * dR, L=self.L)
            Brho, Bz = thin_solenoid.B_elliptic(rho, z)
            return (carry[0] + Brho, carry[1] + Bz), None

        shell_radii = jnp.linspace(self.Rin, self.Rout, num_shells)
        out, _ = jax.lax.scan(
            shell_contrib, (jnp.array(0.0), jnp.array(0.0)), shell_radii
        )
        return out

    def field_strength(
        self, point: Point[Cartesian4]
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        xcyl = point.x.to_cylindric3()
        Brho, Bz = self.B_shells(xcyl.rho, xcyl.z)
        Bphi = jnp.zeros_like(Brho)
        E = Tangent(Point(x=point.x.to_cartesian3()), dx=Cartesian3.make())
        B = Tangent(Point(x=xcyl), dx=Cylindric3.make(rho=Brho, phi=Bphi, z=Bz))
        return E, B.to_cartesian()
