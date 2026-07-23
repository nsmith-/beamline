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
    cylinder radius. Will return positive time if the ray has not reached the cylinder
    or negative time if the ray is inside the cylinder. If the ray has passed the cylinder,
    the returned time will be infinite.

    Args:
        ray: Tangent vector representing the line (e.g. a particle trajectory)
        cyl_point: A point on the cylinder axis
        cyl_axis: The direction of the cylinder axis, normalized to the cylinder radius

    Returns:
        (t, h): Time of intersection and z coordinate of the intersection point

    Following the formulas of https://en.wikipedia.org/wiki/Line-cylinder_intersection
    (with the cylinder radius absorbed into the axis vector for convenience)
    """
    a = cyl_axis
    b = cyl_point - ray.p
    n = ray.t
    # assuming a.a = r^2,
    # ax(n*d - b) . ax(n*d - b) = (a.a)^2
    # d^2 axn . axn - 2d axb . axn + (axb . axb - (a.a)^2) = 0
    # d = (axb . axn  +- sqrt( (axb . axn)^2 - axn.axn * (axb . axb - (a.a)^2) )) / axn . axn
    axn = a.cross(n)
    axb = a.cross(b)
    r2 = a.dot(a)
    tden = axn.dot(axn)
    discriminant = axb.dot(axn) ** 2 - axn.dot(axn) * (axb.dot(axb) - r2 * r2)
    # When tden==0 (ray parallel to axis), discriminant==0 identically, so sqrt'(0)=inf
    # produces a NaN tangent under jax_debug_nans even though the result is masked.
    # Substitute a safe positive value so sqrt is never evaluated at 0.
    safe_disc = jnp.where(tden == 0.0, 1.0, discriminant)
    sd = jnp.where(safe_disc >= 0, jnp.sqrt(jnp.abs(safe_disc)), jnp.inf)
    t1num = axb.dot(axn) + sd
    t2num = axb.dot(axn) - sd
    t = jnp.where(
        tden == 0.0,
        jnp.inf,
        jnp.where(
            # t2 will be the closest if we have not reached the cylinder yet
            t2num >= 0.0,
            t2num,
            jnp.where(
                t1num >= 0.0,
                -t1num,  # t2 < 0, t1 > 0 => inside, so take negative time
                jnp.inf,
            ),
        )
        / jnp.where(tden == 0.0, 1.0, tden),
    )
    # if n has zeros, then inf * 0 causes trouble, so work around it
    h = jnp.where(
        t == jnp.inf,
        jnp.inf,
        a.dot(jnp.where(t == jnp.inf, 0.0, abs(t)) * n - b) / jnp.sqrt(r2),
    )
    return t, h


class Volume(eqx.Module):
    """A volume in space"""

    @abstractmethod
    def contains(self, point: Cartesian3) -> SBool:
        """Whether the point is contained in the volume"""
        raise NotImplementedError

    @abstractmethod
    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        """Signed time to the nearest volume surface

        Returns the smallest positive parametric time ``t`` such that
        ``ray.p + t * ray.t`` lies on a boundary surface, with the sign
        encoding containment:

        - **Positive**: the particle is *outside* the volume; the value is the
          time until it enters (or ``inf`` if the ray never intersects).
        - **Negative**: the particle is *inside* the volume; the magnitude is
          the time until it exits.

        Composite implementations should return the value from whichever
        constituent surface is nearest (smallest absolute time).

        Used for boundary-aware step size control in tracking, as boundaries
        cause discontinuities in the fields or material properties.
        """
        raise NotImplementedError


class CylinderVolume(Volume):
    """Cylinder-shaped volume in space (mixin)"""

    radius: eqx.AbstractVar[SFloat]
    length: eqx.AbstractVar[SFloat]

    def contains(self, point: Cartesian3) -> SBool:
        pcyl = point.to_cylindric()
        return (
            (pcyl.z >= -self.length / 2)
            & (pcyl.z <= self.length / 2)
            & (pcyl.rho <= self.radius)
        )

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # cylindrical side
        tcyl, h = line_cylinder_intersection(
            ray, Cartesian3.make(), Cartesian3.make(z=self.radius)
        )
        tcyl = jnp.where(abs(h) <= self.length / 2, tcyl, jnp.inf)
        # end disks. line_plane_intersection returns (u, v) in units of the
        # basis vectors (here scaled by radius), so the in-disk test is the unit
        # disk u**2 + v**2 <= 1.
        t1, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=-self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=-self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=-self.length / 2),
        )
        t1 = jnp.where(uu**2 + vv**2 <= 1.0, t1, jnp.inf)
        t2, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=self.length / 2),
        )
        t2 = jnp.where(uu**2 + vv**2 <= 1.0, t2, jnp.inf)
        ts = jnp.array([tcyl, t1, t2])
        t_forward = jnp.min(jnp.where(ts >= 0, ts, jnp.inf))
        inside = ((tcyl <= 0.0) | ~jnp.isfinite(tcyl)) & (t1 * t2 <= 0.0)
        return jnp.where(inside, -t_forward, t_forward)
