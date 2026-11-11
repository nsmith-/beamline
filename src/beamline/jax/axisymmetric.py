"""Cartesian fields of axisymmetric sources, smooth through the axis

Going through a cylindrical tangent (``Tangent[Cylindric3].to_cartesian()``) is
singular on the axis: the basis change divides by rho, and rho = hypot(x, y)
has no derivative there. The field itself is smooth through the axis only
because its rho and phi components are odd in rho, which autodiff cannot see.

For a field with no phi dependence, the components F_rho, F_phi are odd in rho
and F_z is even, so

    F_x = x f_rho - y f_phi,    F_y = y f_rho + x f_phi

with f_rho = F_rho / rho and f_phi = F_phi / rho smooth even functions of rho.
Sources provide (f_rho, f_phi, F_z) as functions of rho^2 = x^2 + y^2, which
makes the Cartesian field smooth everywhere, given that they compute them
smoothly in rho^2 (e.g. via a series near the axis).
"""

from collections.abc import Callable
from typing import NamedTuple

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.types import SFloat


class ReducedComponents(NamedTuple):
    """Cylindrical components of a phi-independent vector field

    The odd components are divided by rho, so all three are smooth even
    functions of rho.
    """

    rho: SFloat
    """F_rho / rho"""
    phi: SFloat
    """F_phi / rho"""
    z: SFloat
    """F_z"""


type ReducedFieldFn = Callable[
    [SFloat, SFloat, SFloat], tuple[ReducedComponents, ReducedComponents]
]
"""(rho^2, z, ct) -> (E, B) reduced components"""


def to_cartesian(point: Cartesian4, F: ReducedComponents) -> Tangent[Cartesian3]:
    """Assemble a Cartesian tangent vector from reduced components at point"""
    x, y = point.x, point.y
    return Tangent(
        p=point.to_cartesian3(),
        t=Cartesian3.make(
            x=x * F.rho - y * F.phi,
            y=y * F.rho + x * F.phi,
            z=F.z,
        ),
    )


def axisymmetric_field_strength(
    point: Cartesian4, fields: ReducedFieldFn
) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
    """(E, B) in Cartesian components for an axisymmetric source

    Args:
        point: Point where the field is evaluated
        fields: Reduced cylindrical components of E and B as functions of
            (rho^2, z, ct); see ReducedComponents
    """
    E, B = fields(point.x**2 + point.y**2, point.z, point.ct)
    return to_cartesian(point, E), to_cartesian(point, B)
