from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    DivergenceField,
    GradientField,
    Point,
    Transform,
    TransformOneForm,
)
from beamline.jax.types import SFloat, Vec3


def test_convert_point():
    p_cart = Point(x=Cartesian4(coords=jnp.array([3.0, 0.0, 0.0, 5.0])))
    assert abs(p_cart.x) == 4.0
    p_cyl = Point(x=p_cart.x.to_cylindrical())
    assert abs(p_cyl.x) == 4.0


def potential(p: Point[Cartesian3], origin: Point[Cartesian3] | None = None) -> SFloat:
    if origin is None:
        origin = Point(x=Cartesian3.make())
    r = jnp.sqrt(
        (p.x.x - origin.x.x) ** 2
        + (p.x.y - origin.x.y) ** 2
        + (p.x.z - origin.x.z) ** 2
    )
    return 1 / r


def potentialc(p: Point[Cylindric3]) -> SFloat:
    r = jnp.sqrt(p.x.rho**2 + p.x.z**2)
    return 1 / r


def test_grad():
    """Demonstrate that the gradient field can be computed in either
    coordinate system and converted to the other, yielding the same result.
    """

    field_cart = GradientField(field=potential)
    field_cyl = GradientField(field=potentialc)

    @jax.vmap
    def test_roundtrip(coords: Vec3) -> tuple[Vec3, Vec3]:
        p_cart = Point(x=Cartesian3(coords=coords))
        p_cyl = p_cart.to_cylindrical()

        grad_at_cart = field_cart(p_cart)
        grad_at_cyl = field_cyl(p_cyl)

        cylval = grad_at_cyl.to_cartesian().dx.coords
        cartval = grad_at_cart.dx.coords
        return cylval, cartval

    rng = jax.random.PRNGKey(1234)
    coords = jax.random.uniform(rng, shape=(1000, 3), minval=-5.0, maxval=5.0)
    coords.at[..., :2].set(0.0)  # include some on-axis points
    cylval, cartval = test_roundtrip(coords)
    assert cylval == pytest.approx(cartval, rel=1e-12)


def test_div():
    field_cart = DivergenceField(field=GradientField(potential))
    field_cyl = DivergenceField(field=GradientField(potentialc))

    @jax.vmap
    def test_roundtrip(coords: Vec3) -> tuple[SFloat, SFloat]:
        p_cart = Point(x=Cartesian3(coords=coords))
        p_cyl = p_cart.to_cylindrical()

        div_at_cart = field_cart(p_cart)
        div_at_cyl = field_cyl(p_cyl)

        return div_at_cyl, div_at_cart

    rng = jax.random.PRNGKey(5678)
    coords = jax.random.uniform(rng, shape=(1000, 3), minval=-5.0, maxval=5.0)
    coords.at[::5, :2].set(0.0)  # include some on-axis points
    div_cyl, div_cart = test_roundtrip(coords)
    assert div_cart == pytest.approx(jnp.zeros_like(div_cart), abs=1e-14)
    assert div_cyl == pytest.approx(jnp.zeros_like(div_cart), abs=1e-14)


def monopole_potential(
    origin: Point[Cartesian4], moment: Cartesian3, p: Point[Cartesian4]
) -> SFloat:
    "Something rotationally symmetric"
    r = jnp.sqrt(
        (p.x.x - origin.x.x) ** 2
        + (p.x.y - origin.x.y) ** 2
        + (p.x.z - origin.x.z) ** 2
    )
    return 1 / r


def dipole_potential(
    origin: Point[Cartesian4], moment: Cartesian3, p: Point[Cartesian4]
) -> SFloat:
    "Something not rotationally symmetric"
    pshift = Cartesian4(coords=p.x.coords - origin.x.coords)
    p3 = pshift.to_cartesian3()
    return p3.dot(moment) / abs(p3) ** 3


FType = Callable[[Point[Cartesian4], Cartesian3, Point[Cartesian4]], SFloat]


@pytest.mark.parametrize("func", [monopole_potential, dipole_potential])
def test_transform(func: FType):
    angle = jnp.pi / 3
    origin = Point(x=Cartesian4.make(x=1.0, y=-1.0, z=2.0))
    direction = Cartesian3.make(x=1.0, y=0.0, z=0.0)
    rotated_direction = Cartesian3.make(x=jnp.cos(angle), y=jnp.sin(angle), z=0.0)

    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(x=0.0, y=0.0, z=1.0),
        angle=-angle,
        translation=origin.x,
    )
    field_local = GradientField(
        field=partial(func, Point(Cartesian4.make()), direction)
    )
    field_global = TransformOneForm(transform=transform, field=field_local)
    expected_global = GradientField(field=partial(func, origin, rotated_direction))

    @jax.vmap
    def test_roundtrip(point: Point[Cartesian4]):
        val_global = field_global(point)
        val_expected = expected_global(point)
        return val_global, val_expected

    rng = jax.random.PRNGKey(91011)
    coords = jax.random.uniform(rng, shape=(1000, 4), minval=-5.0, maxval=5.0)
    points = Point(x=Cartesian4(coords=coords))
    val_global, val_expected = test_roundtrip(points)
    assert val_global.point.x.coords == pytest.approx(
        val_expected.point.x.coords, rel=1e-12
    )
    assert val_global.dx.coords == pytest.approx(val_expected.dx.coords, rel=1e-12)
