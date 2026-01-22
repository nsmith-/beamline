"""Simple wire loop model"""

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.elliptic import elliptic_kepi
from beamline.jax.types import SFloat
from beamline.units import MU0


class WireLoop(eqx.Module):
    R: SFloat
    """Radius of the wire loop [mm]"""
    I: SFloat  # noqa: E741
    """Current through the wire loop [e/ns]"""

    def _B(self, rho: SFloat, z: SFloat) -> tuple[SFloat, SFloat]:
        """Magnetic field of a wire loop at (rho, z)

        Args:
            rho: Radial distance from the axis of the loop [mm]
            z: Axial distance from the plane of the loop [mm]

        Returns:
            Tuple containing the radial and axial components of the magnetic field


        References:
            Granum:2022dtk https://doi.org/10.1016/j.nima.2022.166706
            https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html
        """
        squares = self.R**2 + z**2 + rho**2
        alpha2 = squares - 2 * self.R * rho
        beta2 = squares + 2 * self.R * rho
        k2 = 1 - alpha2 / beta2
        C = MU0 * self.I / jnp.pi
        beta = jnp.sqrt(beta2)

        K, E, _ = elliptic_kepi(n=0.0, k=jnp.sqrt(k2))
        Brho = C * z / (2 * alpha2 * beta * rho) * (squares * E - alpha2 * K)
        Bz = C / (2 * alpha2 * beta) * ((self.R**2 - rho**2 - z**2) * E + alpha2 * K)
        return Brho, Bz
