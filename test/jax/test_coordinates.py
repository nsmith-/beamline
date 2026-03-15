from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    DivergenceField,
    GradientField,
    Tangent,
    Transform,
    TransformOneForm,
)
from beamline.jax.types import SFloat


def test_convert_point():
    # TODO: more tests, this is very barebones
    p_cart = Cartesian4(coords=jnp.array([3.0, 0.0, 0.0, 5.0]))
    assert abs(p_cart) == 4.0
    p_cyl = p_cart.to_cylindric()
    assert abs(p_cyl) == 4.0


def test_tangent_conversion():
    tangent_cart = Tangent(
        p=Cartesian3.make(x=2.0),
        t=Cartesian3.make(y=1.0),
    )
    tangent_cyl = tangent_cart.to_cylindric()
    assert tangent_cyl.t.rho == pytest.approx(0.0, rel=1e-15)
    # convention: phihat is normalized so this is 1.0 rather than 0.5
    # (e.g. if t.y is 1 m/s then t.phi is 1 m/s in the phi direction, not 0.5 m/s)
    assert tangent_cyl.t.phi == pytest.approx(1.0, rel=1e-15)
    tangent_cart_back = tangent_cyl.to_cartesian()
    assert tangent_cart_back.t.x == pytest.approx(0.0, rel=1e-15)
    assert tangent_cart_back.t.y == pytest.approx(1.0, rel=1e-15)

    tangent_cart = Tangent(
        p=Cartesian3.make(y=2.0),
        t=Cartesian3.make(x=1.0),
    )
    tangent_cyl = tangent_cart.to_cylindric()
    assert tangent_cyl.t.rho == pytest.approx(0.0, rel=1e-15)
    assert tangent_cyl.t.phi == pytest.approx(-1.0, rel=1e-15)

    def field(p: Cylindric3) -> SFloat:
        return p.y

    grad = GradientField(field=field)(Cartesian3.make(x=2.0).to_cylindric())
    assert grad.t.rho == pytest.approx(0.0, rel=1e-15)
    assert grad.t.phi == pytest.approx(1.0, rel=1e-15)
    grad_cart = grad.to_cartesian()
    assert grad_cart.t.x == pytest.approx(0.0, rel=1e-15)
    assert grad_cart.t.y == pytest.approx(1.0, rel=1e-15)


def potential(p: Cartesian3, origin: Cartesian3 | None = None) -> SFloat:
    if origin is None:
        origin = Cartesian3.make()
    r = jnp.sqrt((p.x - origin.x) ** 2 + (p.y - origin.y) ** 2 + (p.z - origin.z) ** 2)
    return 1 / r


def potential_cyl(p: Cylindric3) -> SFloat:
    r = jnp.sqrt(p.rho**2 + p.z**2)
    return 1 / r


def potential_cyl_displaced(p: Cylindric3, origin: Cartesian3) -> SFloat:
    p_cart = p.to_cartesian()
    return potential(p_cart, origin=origin)


def test_grad():
    """Demonstrate that the gradient field can be computed in either
    coordinate system and converted to the other, yielding the same result.
    """

    field_cart = GradientField(field=potential)
    field_cyl = GradientField(field=potential_cyl)

    shift = Cartesian3.make(x=3.0, y=2.0, z=1.0)
    field_displaced = GradientField(partial(potential, origin=shift))
    field_cyl_displaced = GradientField(partial(potential_cyl_displaced, origin=shift))

    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(z=1.0),
        angle=0.0,
        translation=Cartesian4(jnp.zeros(4).at[:3].set(shift.coords)),
    )
    field_transform = TransformOneForm(transform=transform, field=field_cart)

    rng = jax.random.PRNGKey(1234)
    coords = jax.random.uniform(rng, shape=(1000, 3), minval=-5.0, maxval=5.0)
    # coords = coords.at[::5, :2].set(0.0)  # include some on-axis points
    input = Cartesian3(coords=coords)

    grad_cart = jax.vmap(field_cart)(input)
    grad_cyl = jax.vmap(field_cyl)(input.to_cylindric())
    grad_cart_displaced = jax.vmap(field_displaced)(input)
    grad_cyl_displaced = jax.vmap(field_cyl_displaced)(input.to_cylindric())
    grad_cart_transform = jax.vmap(field_transform)(input)
    assert grad_cart.t.coords == pytest.approx(
        grad_cyl.to_cartesian().t.coords, rel=1e-14
    )
    assert grad_cart_transform.t.coords == pytest.approx(
        grad_cart_displaced.t.coords, rel=1e-14
    )
    assert grad_cart_displaced.to_cylindric().t.coords == pytest.approx(
        grad_cyl_displaced.t.coords, rel=1e-14
    )
    assert grad_cart_displaced.t.coords == pytest.approx(
        grad_cyl_displaced.to_cartesian().t.coords, rel=1e-14
    )


def test_div():
    field_cart = DivergenceField(field=GradientField(potential))
    field_cyl = DivergenceField(field=GradientField(potential_cyl))

    shift = Cartesian3.make(x=3.0, y=2.0, z=1.0)
    field_displaced = DivergenceField(
        field=GradientField(partial(potential, origin=shift))
    )
    field_cyl_displaced = DivergenceField(
        field=GradientField(partial(potential_cyl_displaced, origin=shift))
    )

    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(z=1.0),
        angle=0.0,
        translation=Cartesian4(jnp.zeros(4).at[:3].set(shift.coords)),
    )
    field_transform = DivergenceField(
        field=TransformOneForm(transform=transform, field=GradientField(potential))
    )

    rng = jax.random.PRNGKey(5678)
    coords = jax.random.uniform(rng, shape=(1000, 3), minval=-5.0, maxval=5.0)
    # TODO: make this work at zero
    # coords = coords.at[::5, :2].set(0.0)  # include some on-axis points
    input = Cartesian3(coords=coords)

    div_cart = jax.vmap(field_cart)(input)
    div_cyl = jax.vmap(field_cyl)(input.to_cylindric())
    div_displaced = jax.vmap(field_displaced)(input)
    div_cyl_displaced = jax.vmap(field_cyl_displaced)(input.to_cylindric())
    div_cart_transform = jax.vmap(field_transform)(input)
    zero = pytest.approx(jnp.zeros_like(div_cart), abs=2e-14)
    assert div_cart == zero
    assert div_cyl == zero
    assert div_displaced == zero
    assert div_cyl_displaced == zero
    assert div_cart_transform == zero


def test_transform_rotate():
    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(z=1.0),
        angle=jnp.pi / 2,
        translation=Cartesian4.make(),
    )

    vec = Tangent(
        p=Cartesian4.make(),
        t=Cartesian4.make(y=1.0),
    )
    vec_loc = transform.tangent_to_local(vec)
    vec_loc_exp = Tangent(
        p=Cartesian4.make(),
        t=Cartesian4.make(x=1.0),
    )
    assert vec_loc.p.coords == pytest.approx(vec_loc_exp.p.coords, rel=1e-12)
    assert vec_loc.t.coords == pytest.approx(vec_loc_exp.t.coords, rel=1e-12)

    vec = Tangent(
        p=Cartesian4.make(),
        t=Cartesian4.make(z=1.0),
    )
    vec_loc = transform.tangent_to_local(vec)
    vec_loc_exp = Tangent(
        p=Cartesian4.make(),
        t=Cartesian4.make(z=1.0),
    )
    assert vec_loc.p.coords == pytest.approx(vec_loc_exp.p.coords, rel=1e-12)
    assert vec_loc.t.coords == pytest.approx(vec_loc_exp.t.coords, rel=1e-12)

    vec = Tangent(
        p=Cartesian4.make(z=1.0),
        t=Cartesian4.make(y=1.0),
    )
    vec_loc = transform.tangent_to_local(vec)
    vec_loc_exp = Tangent(
        p=Cartesian4.make(z=1.0),
        t=Cartesian4.make(x=1.0),
    )
    assert vec_loc.p.coords == pytest.approx(vec_loc_exp.p.coords, rel=1e-12)
    assert vec_loc.t.coords == pytest.approx(vec_loc_exp.t.coords, rel=1e-12)

    vec = Tangent(
        p=Cartesian4.make(x=1.0),
        t=Cartesian4.make(y=1.0),
    )
    vec_loc = transform.tangent_to_local(vec)
    vec_loc_exp = Tangent(
        p=Cartesian4.make(y=-1.0),
        t=Cartesian4.make(x=1.0),
    )
    assert vec_loc.p.coords == pytest.approx(vec_loc_exp.p.coords, rel=1e-12)
    assert vec_loc.t.coords == pytest.approx(vec_loc_exp.t.coords, rel=1e-12)


def test_transform_both():
    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(z=1.0),
        angle=jnp.pi / 2,
        translation=Cartesian4.make(x=3.0),
    )
    vec = Tangent(
        p=Cartesian4.make(x=1.0),
        t=Cartesian4.make(y=1.0),
    )
    vec_loc = transform.tangent_to_local(vec)
    vec_loc_exp = Tangent(
        p=Cartesian4.make(x=0.0, y=2.0),
        t=Cartesian4.make(x=1.0),
    )
    assert vec_loc.p.coords == pytest.approx(vec_loc_exp.p.coords, rel=1e-12)
    assert vec_loc.t.coords == pytest.approx(vec_loc_exp.t.coords, rel=1e-12)


def monopole_potential(origin: Cartesian4, moment: Cartesian3, p: Cartesian4) -> SFloat:
    "Something rotationally symmetric"
    r = jnp.sqrt((p.x - origin.x) ** 2 + (p.y - origin.y) ** 2 + (p.z - origin.z) ** 2)
    return 1 / r


def dipole_potential(origin: Cartesian4, moment: Cartesian3, p: Cartesian4) -> SFloat:
    "Something not rotationally symmetric"
    # TODO: __sub__ mixin
    pshift = Cartesian4(coords=p.coords - origin.coords)
    p3 = pshift.to_cartesian3()
    return p3.dot(moment) / abs(p3) ** 3


FType = Callable[[Cartesian4, Cartesian3, Cartesian4], SFloat]


@pytest.mark.parametrize("func", [monopole_potential, dipole_potential])
def test_transform_tangents(func: FType):
    angle = jnp.pi / 3
    origin = Cartesian4.make(x=1.0, y=-1.0, z=2.0)
    direction = Cartesian3.make(x=1.0, y=0.0, z=0.0)
    rotated_direction = Cartesian3.make(x=jnp.cos(angle), y=jnp.sin(angle), z=0.0)

    transform = Transform.make_axis_angle(
        axis=Cartesian3.make(x=0.0, y=0.0, z=1.0),
        angle=angle,
        translation=origin,
    )
    field_local = GradientField(field=partial(func, Cartesian4.make(), direction))
    field_global = TransformOneForm(transform=transform, field=field_local)
    expected_global = GradientField(field=partial(func, origin, rotated_direction))

    @jax.vmap
    def test_roundtrip(point: Cartesian4):
        val_global = field_global(point)
        val_expected = expected_global(point)
        return val_global, val_expected

    rng = jax.random.PRNGKey(91011)
    coords = jax.random.uniform(rng, shape=(1000, 4), minval=-5.0, maxval=5.0)
    points = Cartesian4(coords=coords)
    val_global, val_expected = test_roundtrip(points)
    assert val_global.p.coords == pytest.approx(val_expected.p.coords, rel=1e-12)
    assert val_global.t.coords == pytest.approx(val_expected.t.coords, rel=1e-12)


def test_pretty_dipole(artifacts_dir: Path, request):
    """Generate a plot of the dipole potential and its gradient field"""

    origin = Cartesian4.make()
    moment = Cartesian3.make(x=0.2, y=1.0, z=1.0)

    local_field = GradientField(field=partial(dipole_potential, origin, moment))

    x = jnp.linspace(-2.0, 2.0, 30)
    z = jnp.linspace(-2.0, 2.0, 30)
    X, Z = jnp.meshgrid(x, z)
    X = X.ravel()
    Z = Z.ravel()
    Y = jnp.zeros_like(X)
    cut = jnp.hypot(X, Z) > 0.3
    X = X[cut]
    Y = Y[cut]
    Z = Z[cut]

    @jax.jit
    def get_data(angle):
        @jax.vmap
        def fieldvals(x, y, z):
            global_field = TransformOneForm(
                transform=Transform.make_axis_angle(
                    axis=Cartesian3.make(x=1.0),
                    angle=angle,
                    translation=Cartesian4.make(),
                ),
                field=local_field,
            )
            return global_field(Cartesian4.make(x=x, y=y, z=z))

        vals = fieldvals(X, Y, Z)
        dx = vals.t.x
        dz = vals.t.z
        norm = jnp.sqrt(jnp.sum(vals.t.coords[..., :3] ** 2, axis=-1))
        # ad-hoc sqrt rescaling to make arrows more visible
        dx = dx / jnp.sqrt(norm)
        dz = dz / jnp.sqrt(norm)
        return dx, dz

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title("Dipole Potential Gradient Field")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    dx, dz = get_data(0.0)
    quiver = ax.quiver(
        X, Z, dx, dz, angles="xy", scale_units="xy", pivot="middle", scale=1e1
    )

    nframes = 40

    def update(frame: int):
        angle = frame * (2 * jnp.pi / nframes)
        dx, dz = get_data(angle)
        quiver.set_UVC(dx, dz)
        return (quiver,)

    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        blit=True,
    )
    anim.save(artifacts_dir / f"{request.node.name}.gif", writer="pillow", fps=10)
