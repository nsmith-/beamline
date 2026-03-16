import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import line_cylinder_intersection, line_plane_intersection


def approx(val):
    return pytest.approx(val, rel=1e-15, abs=1e-15)


def test_line_plane_intersection():
    pcenter = Cartesian3.make()
    pu = Cartesian3.make(x=1.0)
    pv = Cartesian3.make(y=1.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == approx(1.0)
    assert u == approx(0.0)
    assert v == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0, z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == approx(1.0)
    assert u == approx(1.0)
    assert v == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert jnp.isnan(t)
    assert u == jnp.inf
    assert v == -jnp.inf  # must be an artifact of the algorithm


def test_line_plane_intersection_grad():
    def ray_plane(plane_z):
        pcenter = Cartesian3.make(z=plane_z)
        pu = Cartesian3.make(x=1.0, z=plane_z)
        pv = Cartesian3.make(y=1.0, z=plane_z)
        ray = Tangent(
            p=Cartesian3.make(z=-1.0),
            t=Cartesian3.make(x=1.0, z=1.0),
        )
        t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
        return t, u, v

    t, u, v = ray_plane(0.0)
    assert t == approx(1.0)
    assert u == approx(1.0)
    assert v == approx(0.0)

    dt_dz, du_dz, dv_dz = jax.jacfwd(ray_plane)(0.0)
    assert dt_dz == approx(1.0)
    assert du_dz == approx(1.0)
    assert dv_dz == approx(0.0)

    dt_dz, du_dz, dv_dz = jax.jacrev(ray_plane)(0.0)
    assert dt_dz == approx(1.0)
    assert du_dz == approx(1.0)
    assert dv_dz == approx(0.0)

    # TODO: test more exotic cases


def test_line_cylinder_intersection():
    cyl_point = Cartesian3.make()
    cyl_axis = Cartesian3.make(x=1.0)

    ray = Tangent(
        p=Cartesian3.make(y=-1.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(0.0)
    assert h == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(z=0.1),
        t=Cartesian3.make(z=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(0.9)
    assert h == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=2.0, y=2.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(0.5)
    assert h == approx(1.0)

    ray = Tangent(
        p=Cartesian3.make(x=0.1, y=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == jnp.inf
    assert h == jnp.inf

    cyl_axis = Cartesian3.make(z=3.0)
    ray = Tangent(
        p=Cartesian3.make(y=0.5, z=1.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(2.5)
    assert h == approx(1.0)


def test_line_cylinder_intersection_grad():
    def ray_cylinder(cyl_radius):
        cyl_point = Cartesian3.make()
        cyl_axis = Cartesian3.make(x=cyl_radius)
        ray = Tangent(
            p=Cartesian3.make(y=0.25),
            t=Cartesian3.make(y=1.0),
        )
        t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
        return t, h

    t, h = ray_cylinder(1.0)
    assert t == approx(0.75)
    assert h == approx(0.0)

    dt_dr, dh_dr = jax.jacfwd(ray_cylinder)(1.0)
    assert dt_dr == approx(1.0)
    assert dh_dr == approx(0.0)

    dt_dr, dh_dr = jax.jacrev(ray_cylinder)(1.0)
    assert dt_dr == approx(1.0)
    assert dh_dr == approx(0.0)

    def ray_cylinder(cyl_x):
        cyl_point = Cartesian3.make(x=cyl_x)
        cyl_axis = Cartesian3.make(x=1.0)
        ray = Tangent(
            p=Cartesian3.make(x=-1.0),
            t=Cartesian3.make(x=1.0),
        )
        t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
        return t, h

    t, h = ray_cylinder(0.0)
    assert t == jnp.inf
    assert h == jnp.inf

    return pytest.xfail(
        reason="When parallel, the auto-diff is not handling inf correctly"
    )
    # TODO: is grad 0 the right behavior?

    dt_dx, dh_dx = jax.jacfwd(ray_cylinder)(0.0)
    assert dt_dx == 0.0
    assert dh_dx == 0.0

    dt_dx, dh_dx = jax.jacrev(ray_cylinder)(0.0)
    assert dt_dx == 0.0
    assert dh_dx == 0.0
