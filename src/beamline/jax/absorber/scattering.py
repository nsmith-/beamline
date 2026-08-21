"""Multiple Coulomb scattering sampling (PDG 34.3, 34.22)

Highland: two independent projected planes, each with RMS ``theta0`` (34.16). 
Within each plane the exit angle and the in-material lateral offset are
correlated (rho = sqrt(3)/2) and drawn jointly from two standard normals (34.22).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from beamline.jax.absorber.material import InteractionParams
from beamline.jax.types import SFloat


def highland_scattering_sampler(
    params: InteractionParams, key: Array
) -> tuple[SFloat, SFloat, SFloat, SFloat]:
    """Sample correlated (angle, offset) pairs in two planes.

    For each plane, with two standard normals z1, z2:
        theta = z2 * theta0
            y = z1 * (x*theta0/sqrt(12)) + z2 * (x*theta0/2)
    so angle and offset share z2. ``params`` supplies theta0 and the segment
    thickness x.

    Returns:
        (theta_x, theta_y, y_x, y_y).
    """
    theta0 = params.theta0
    x = params.thickness
    kx, ky = jr.split(key)
    k1x, k2x = jr.split(kx)
    k1y, k2y = jr.split(ky)
    z1x, z2x = jr.normal(k1x), jr.normal(k2x)
    z1y, z2y = jr.normal(k1y), jr.normal(k2y)
    a = x * theta0 / jnp.sqrt(12.0)
    b = x * theta0 / 2.0
    theta_x, theta_y = z2x * theta0, z2y * theta0
    y_x = z1x * a + z2x * b
    y_y = z1y * a + z2y * b
    return theta_x, theta_y, y_x, y_y
