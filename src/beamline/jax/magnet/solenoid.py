"""Solenoid models"""

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.elliptic import elliptic_kepi
from beamline.jax.types import SFloat
from beamline.units import to_clhep, ureg

MU0 = to_clhep(1 * ureg.vacuum_permeability)


def _auxp12(k2, gamma):
    """Auxiliary functions for implementing 10.1109/20.947050

    From Eqn. 4, 6"""
    sqrt1mk2 = jnp.sqrt(1 - k2)
    _1mgamma2 = 1 - gamma**2
    K, E, Pi = elliptic_kepi(_1mgamma2, sqrt1mk2)
    P1 = K - 2 * (K - E) / (1 - k2)
    P2 = -gamma / _1mgamma2 * (Pi - K) - 1 / _1mgamma2 * (gamma**2 * Pi - K)
    return P1, P2


class ThinShellSolenoid(eqx.Module):
    R: SFloat
    """Shell radius [mm]"""
    jphi: SFloat
    """Surface current density [e/ns]"""
    L: SFloat
    """Length of the solenoid [mm]"""

    def _B_Caciagli(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Return the rho and z component of the magnetic field

        Using the exact solution of 10.1016/j.nima.2022.166706
        TODO: check if this is actually less expensive than the Conway solution of 10.1109/20.947050

        This formula is ill-defined for rho=0, so we clamp rho to a small value
        """
        rho = jnp.maximum(rho, 1e-7)
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
