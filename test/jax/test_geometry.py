import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import line_cylinder_intersection, line_plane_intersection


def test_line_plane_intersection():
    pcenter = Cartesian3.make()
    pu = Cartesian3.make(x=1.0)
    pv = Cartesian3.make(y=1.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == pytest.approx(1.0, rel=1e-15)
    assert u == pytest.approx(0.0, rel=1e-15)
    assert v == pytest.approx(0.0, rel=1e-15)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0, z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == pytest.approx(1.0, rel=1e-15)
    assert u == pytest.approx(1.0, rel=1e-15)
    assert v == pytest.approx(0.0, rel=1e-15)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert jnp.isnan(t)
    assert u == jnp.inf
    assert v == -jnp.inf  # must be an artifact of the algorithm


def test_line_cylinder_intersection():
    cyl_point = Cartesian3.make()
    cyl_axis = Cartesian3.make(x=1.0)

    ray = Tangent(
        p=Cartesian3.make(y=-1.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == pytest.approx(0.0, abs=1e-15)
    assert h == pytest.approx(0.0, rel=1e-15)

    ray = Tangent(
        p=Cartesian3.make(z=0.1),
        t=Cartesian3.make(z=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == pytest.approx(0.9, rel=1e-15)
    assert h == pytest.approx(0.0, rel=1e-15)

    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=2.0, y=2.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == pytest.approx(0.5, rel=1e-15)
    assert h == pytest.approx(1.0, rel=1e-15)

    ray = Tangent(
        p=Cartesian3.make(x=0.1, y=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == jnp.inf
    assert h == jnp.inf
