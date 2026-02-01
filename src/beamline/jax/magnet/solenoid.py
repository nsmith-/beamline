"""Solenoid models"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.scipy.special import gamma
from quadax import quadgk

from beamline.jax.elliptic import (
    elliprd_one_zero,
    elliprf_one_zero,
    elliprj,
    elliptic_kepi,
)
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.types import SFloat
from beamline.units import MU0


def _hypot_ratio(x: SFloat, y: SFloat) -> SFloat:
    """Calculate x/sqrt(x**2+y**2)"""
    hypot = jnp.hypot(x, y)
    return x / hypot


def _auxp12(k2: SFloat, gamma: SFloat) -> tuple[SFloat, SFloat]:
    """Auxiliary functions for implementing 10.1109/20.947050

    From Eqn. 4, 6

    Args:
        k2: Real value in (0, 1], k2 = 1 corresponds to rho = 0
        gamma: Real value in [-1, 1), gamma = -1 corresponds to rho = 0
    """
    # sqrt1mk2 = jnp.sqrt(1 - k2)  # [0, 1)
    # _1mgamma2 = 1 - gamma**2  # [0, 1]
    # K, E, Pi = elliptic_kepi(n=_1mgamma2, k=sqrt1mk2)
    # P1 = K - 2 * (K - E) / (1 - k2)
    # P2 = -gamma / _1mgamma2 * (Pi - K) - 1 / _1mgamma2 * (gamma**2 * Pi - K)
    # Alternative formulation using Carlson integrals (more stable numerically at rho = 0)
    # K = Rf
    # E = Rf - (1-k2) / 3 * Rd
    # Pi = Rf + (1 - gamma**2) / 3 * Rj
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


class ThinShellSolenoid(eqx.Module):
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

    def B_Caciagli(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Using Caciagli Eqn. 3-6
        Note this solution is nan for rho=0

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

    def B_composite(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Uses both a closed-form solution (Caciagli) and a series expansion
        in rho around the on-axis field, as the derivative of the closed-form solution
        becomes a bit inaccurate for small rho.
        """
        rel = 1e-7  # optimized in test_optimize_rho0limit
        Brho_lo, Bz_lo = self.B_rhoexpansion(rho, z, order=2)
        Brho_hi, Bz_hi = self.B_Caciagli(rho, z)
        Brho = jax.lax.select(rho < rel * self.R, Brho_lo, Brho_hi)
        Bz = jax.lax.select(rho < rel * self.R, Bz_lo, Bz_hi)
        return Brho, Bz
