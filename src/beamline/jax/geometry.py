"""Geometry abstractions"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.types import SBool, SFloat


def line_plane_intersection(
    ray: Tangent[Cartesian3],
    plane_point: Cartesian3,
    plane_u: Cartesian3,
    plane_v: Cartesian3,
) -> tuple[SFloat, SFloat, SFloat]:
    """Line-plane intersection

    Computes the time and u, v coordinates of the intersection of a vector
    with a plane defined by a point and two basis vectors. Uses parametric
    form to capture plane coordinates of the intersection, useful for checking
    if inside some boundary in the plane.
    https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Parametric_form

    Args:
        ray: Tangent vector representing the line (e.g. a particle trajectory)
        plane_point: A point on the plane
        plane_u: A point one unit away from plane_point in the u direction of the plane
        plane_v: A point one unit away from plane_point in the v direction of the plane

    Returns:
        (t, u, v): Time of intersection and plane coordinates of the intersection point
    """
    p01 = plane_u - plane_point
    p02 = plane_v - plane_point
    lmp = ray.p - plane_point
    # lmp = -vec.t * t + p01 * u + p02 * v
    mat = jnp.stack([-ray.t.coords, p01.coords, p02.coords], axis=-1)
    rhs = lmp.coords
    sol = jnp.linalg.solve(mat, rhs)
    t, u, v = sol[..., 0], sol[..., 1], sol[..., 2]
    return t, u, v


def line_cylinder_intersection(
    ray: Tangent[Cartesian3], cyl_point: Cartesian3, cyl_axis: Cartesian3
) -> tuple[SFloat, SFloat]:
    """Line-cylinder intersection

    Computes the time and azimuthal coordinate of the intersection of a vector
    with an infinite cylinder defined by a point and axis. The axis magnitude is the
    cylinder radius. Will return the closest intersetion point, either in the future
    or the past.

    Args:
        ray: Tangent vector representing the line (e.g. a particle trajectory)
        cyl_point: A point on the cylinder axis
        cyl_axis: The direction of the cylinder axis, normalized to the cylinder radius

    Returns:
        (t, h): Time of intersection and z coordinate of the intersection point

    Following the formulas of https://en.wikipedia.org/wiki/Line-cylinder_intersection
    """
    a = cyl_axis
    b = cyl_point - ray.p
    n = ray.t
    nxa = n.cross(a)
    a2 = a.dot(a)
    discriminant = nxa.dot(nxa) - a2 * (b.dot(nxa)) ** 2
    sd = jnp.where(discriminant >= 0, jnp.sqrt(jnp.abs(discriminant)), jnp.inf)
    t1num = nxa.dot(b.cross(a)) + sd
    t2num = nxa.dot(b.cross(a)) - sd
    tden = nxa.dot(nxa)
    t = jnp.where(
        tden == 0.0,
        jnp.inf,
        jnp.where(abs(t1num) <= abs(t2num), t1num, t2num)
        / jnp.where(tden == 0.0, 1.0, tden),
    )
    # if vec.t has zeros, then inf * 0 causes trouble, so work around it
    h = jnp.where(
        t == jnp.inf,
        jnp.inf,
        a.dot(jnp.where(t == jnp.inf, 0.0, t) * ray.t - cyl_point) / a2,
    )
    return t, h


class Volume(eqx.Module):
    """A volume in space"""

    @abstractmethod
    def contains(self, point: Cartesian3) -> SBool:
        """Whether the point is contained in the volume"""
        raise NotImplementedError

    @abstractmethod
    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        """Signed time to the volume surface

        This will be the time to the surface, assuming the ray tangent measures
        the speed, with the sign indicating whether the intersection is in the
        future (positive) or past (negative). If there is no intersection, this
        should return inf.  When this method is implemented on composite
        volumes, it should return the minimum signed distance to any of the
        constituents.

        This is useful for efficient step size control in tracking, as
        boundaries may cause discontinuities in the fields or material
        properties.

        """
        raise NotImplementedError
