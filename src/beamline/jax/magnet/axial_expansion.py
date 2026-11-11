"""Helpers for axisymmetric magnetic fields near the axis"""

from collections.abc import Callable

import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.scipy.special import gamma

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.types import SFloat

AXIS_SERIES_RADIUS = 1e-3
"""Radius, relative to the source size, below which axisymmetric_field_strength
uses the rho expansion instead of the full field"""
AXIS_SERIES_ORDER = 2
"""Order of the rho expansion used by axisymmetric_field_strength"""


def _rhoexpansion_g(
    Bz_onaxis: Callable[[SFloat], SFloat], rho2: SFloat, z: SFloat, order: int
) -> tuple[SFloat, SFloat]:
    """(Brho / rho, Bz) from the rho expansion, as functions of rho^2

    Both are polynomials in rho^2, so this is smooth through the axis.
    """
    Bz0, dnBz = jet(
        Bz_onaxis,
        primals=(z,),
        series=(jnp.zeros(2 * order + 1).at[0].set(1.0),),
    )
    q = rho2 / 4
    Bz = Bz0 + sum(
        (-1) ** n / gamma(n + 1) ** 2 * q**n * dnBz[2 * n - 1]
        for n in range(1, order + 1)
    )
    g = sum(
        (-1) ** (n + 1) / (n + 1) / gamma(n + 1) ** 2 / 2 * q**n * dnBz[2 * n]
        for n in range(order + 1)
    )
    return g, Bz


def B_rhoexpansion(
    Bz_onaxis: Callable[[SFloat], SFloat], rho: SFloat, z: SFloat, order: int = 1
) -> tuple[SFloat, SFloat]:
    """Field of an axisymmetric source from a series expansion in rho

    McDonald model: expands in powers of rho around the on-axis field Bz_onaxis(z),
    using its z derivatives up to order 2 * order.

    Returns:
        (Brho, Bz)
    """
    g, Bz = _rhoexpansion_g(Bz_onaxis, rho**2, z, order)
    return rho * g, Bz


def axisymmetric_field_strength(
    B: Callable[[SFloat, SFloat], tuple[SFloat, SFloat]],
    Bz_onaxis: Callable[[SFloat], SFloat],
    point: Cartesian4,
    rho_series: SFloat,
    order: int = AXIS_SERIES_ORDER,
) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
    """Cartesian (E, B) of a static axisymmetric magnetic field

    Going through a cylindrical tangent is singular on the axis: the basis
    change divides by rho, and rho = hypot(x, y) has no derivative there. The
    field itself is smooth through the axis only because Brho is odd in rho,
    which autodiff cannot see.

    Instead, write Bx = x g, By = y g with g = Brho / rho. Within rho_series of
    the axis, g and Bz come from the rho expansion of Bz_onaxis, which is a
    polynomial in rho^2 = x^2 + y^2 and so exact (to truncation) and smooth
    through the axis. Outside, they come from the full field B. With
    rho_series = 1e-3 L (AXIS_SERIES_RADIUS) and order 2 (AXIS_SERIES_ORDER),
    where L is the source size, the two agree to ~1e-12 at the switch. Much
    smaller radii make the jump larger, as Brho from the full field loses
    digits to cancellation near the axis.

    Both branches are evaluated (as with any jnp.where), so this costs one extra
    jet of Bz_onaxis per call.

    Args:
        B: Full field in cylindrical components, (rho, z) -> (Brho, Bz)
        Bz_onaxis: On-axis field, z -> Bz
        point: Point where the field is evaluated
        rho_series: Radius below which the rho expansion is used [mm]
        order: Order of the rho expansion
    """
    x, y, z = point.x, point.y, point.z
    rho2 = x**2 + y**2
    near = rho2 < rho_series**2
    g_series, Bz_series = _rhoexpansion_g(Bz_onaxis, rho2, z, order)
    # double-where: keep the full-field branch (and its gradient) away from rho = 0
    rho = jnp.sqrt(jnp.where(near, rho_series**2, rho2))
    Brho_full, Bz_full = B(rho, z)
    g = jnp.where(near, g_series, Brho_full / rho)
    Bz = jnp.where(near, Bz_series, Bz_full)
    p = point.to_cartesian3()
    E = Tangent(p=p, t=Cartesian3.make())
    Bvec = Tangent(p=p, t=Cartesian3.make(x=g * x, y=g * y, z=Bz))
    return E, Bvec
