"""Multiple Coulomb scattering sampling (PDG 34.3)

The Highland approximation: two independent Gaussian deflection angles in
orthogonal planes containing the particle direction, each with RMS ``theta0``
(PDG 34.16, 34.18).

Only angles are sampled. The correlated lateral offset of PDG 34.22 is not
applied here: the integrator propagates position from the deflected direction
and generates that displacement itself. The resulting quadrature deficit versus
single-application Highland is characterised in test_absorber.py.
"""

from __future__ import annotations

import jax.random as jr
from jax import Array

from beamline.jax.absorber.material import InteractionParams
from beamline.jax.types import SFloat


def highland_scattering_sampler(
    params: InteractionParams, key: Array
) -> tuple[SFloat, SFloat]:
    """Sample projected deflection angles (theta_x, theta_y) [rad]

    Two independent zero-mean Gaussians of width params.theta0. Meant to
    replicate the (params, key) -> sample shape of the straggling samplers.

    Args:
        params: Interaction parameters (uses ``theta0``).
        key: A JAX PRNG key.

    Returns:
        ``(theta_x, theta_y)``, the deflections in two orthogonal planes.
    """
    kx, ky = jr.split(key)
    return params.theta0 * jr.normal(kx), params.theta0 * jr.normal(ky)
