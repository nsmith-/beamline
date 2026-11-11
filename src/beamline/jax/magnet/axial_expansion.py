"""Off-axis field of an axisymmetric source from its on-axis field"""

from collections.abc import Callable

import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.scipy.special import gamma

from beamline.jax.types import SFloat


def B_rhoexpansion(
    Bz_onaxis: Callable[[SFloat], SFloat], rho: SFloat, z: SFloat, order: int = 1
) -> tuple[SFloat, SFloat]:
    """Field of an axisymmetric source from a series expansion in rho

    McDonald model: expands in powers of rho around the on-axis field Bz_onaxis(z),
    using its z derivatives up to order 2 * order.

    Returns:
        (Brho, Bz)
    """
    Bz0, dnBz = jet(
        Bz_onaxis,
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
