"""Coordinate systems implemented in equinox

Note: we try to implement everything in a broadcast-friendly way, so all
coordinates are arrays of shape (..., N), where N is the dimension of the
coordinate system (3 for spatial, 4 for spacetime). The leading dimensions
are for broadcasting over multiple points/vectors.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Self, overload

import equinox as eqx
import jax
import jax.numpy as jnp

from beamline.jax.types import Rot4Matrix, SFloat, Vec3, Vec4, VecN


class CoordinateChart[T: VecN](eqx.Module):
    """Base class for coordinate charts"""

    coords: eqx.AbstractVar[T]

    @abstractmethod
    def to_cartesian(self) -> Cartesian:
        """Convert to Cartesian coordinates"""

    @abstractmethod
    def to_cylindrical(self) -> Cylindric:
        """Convert to Cylindrical coordinates"""

    @abstractmethod
    def __abs__(self) -> SFloat:
        """Norm of the vector"""

    @abstractmethod
    def differential(self) -> T:
        """Differential element in this coordinate chart"""

    @abstractmethod
    def volume_element(self) -> SFloat:
        """Volume element in this coordinate chart"""


class Cylindric(CoordinateChart):
    def to_cylindrical(self) -> Self:
        return self


class Cartesian(CoordinateChart):
    def to_cartesian(self) -> Self:
        return self


class XYMixin:
    coords: Vec3 | Vec4

    @property
    def x(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def y(self) -> SFloat:
        return self.coords[..., 1]


class PolarMixin:
    coords: Vec3 | Vec4

    @property
    def rho(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def phi(self) -> SFloat:
        return self.coords[..., 1]


class ZMixin:
    coords: Vec3 | Vec4

    @property
    def z(self) -> SFloat:
        return self.coords[..., 2]


class TimeMixin:
    coords: Vec4

    @property
    def ct(self) -> SFloat:
        return self.coords[..., 3]


def delta_phi(phi1: SFloat, phi2: SFloat) -> SFloat:
    """Compute (phi1-phi2) wrapped to [-pi, pi)"""
    dphi = phi1 - phi2
    return (dphi + jnp.pi) % (2 * jnp.pi) - jnp.pi


class Cylindric3(Cylindric, PolarMixin, ZMixin):
    coords: Vec3
    """Cylindrical coordinates (rho, phi, z) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0
    ) -> Cylindric3:
        """Create Cylindric3 from individual components"""
        return cls(coords=jnp.stack([rho, phi, z], axis=-1))

    def to_cartesian(self) -> Cartesian3:
        x = self.rho * jnp.cos(self.phi)
        y = self.rho * jnp.sin(self.phi)
        return Cartesian3(coords=jnp.stack([x, y, self.z], axis=-1))

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.rho**2 + self.z**2)

    def differential(self) -> Vec3:
        # drho, rho dphi, dz
        return jnp.ones_like(self.coords).at[..., 1].set(self.rho)

    def volume_element(self) -> SFloat:
        return self.rho


class Cartesian3(Cartesian, XYMixin, ZMixin):
    coords: Vec3
    """Cartesian coordinates (x, y, z) [mm]"""

    @classmethod
    def make(cls, *, x: SFloat = 0.0, y: SFloat = 0.0, z: SFloat = 0.0) -> Cartesian3:
        """Create Cartesian3 from individual components"""
        return cls(coords=jnp.stack([x, y, z], axis=-1))

    def to_cylindrical(self) -> Cylindric3:
        rho = jnp.hypot(self.x, self.y)
        phi = jnp.arctan2(self.y, self.x)
        return Cylindric3(coords=jnp.stack([rho, phi, self.z], axis=-1))

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)

    def differential(self) -> Vec3:
        return jnp.ones_like(self.coords)

    def volume_element(self) -> SFloat:
        return jnp.ones_like(self.x)

    def dot(self, other: Cartesian3) -> SFloat:
        return jnp.sum(self.coords * other.coords, axis=-1)


class Cylindric4(Cylindric, PolarMixin, ZMixin, TimeMixin):
    coords: Vec4
    """Cylindrical coordinates (rho, phi, z, ct) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0, ct: SFloat = 0.0
    ) -> Cylindric4:
        """Create Cylindric4 from individual components"""
        return cls(coords=jnp.stack([rho, phi, z, ct], axis=-1))

    def to_cartesian(self) -> Cartesian4:
        x = self.rho * jnp.cos(self.phi)
        y = self.rho * jnp.sin(self.phi)
        return Cartesian4(coords=jnp.stack([x, y, self.z, self.ct], axis=-1))

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.ct**2 - self.rho**2 - self.z**2)

    def differential(self) -> Vec4:
        # TODO: correct for metric?
        return jnp.ones_like(self.coords).at[..., 1].set(self.rho)

    def volume_element(self) -> SFloat:
        return self.rho


class Cartesian4(Cartesian, XYMixin, ZMixin, TimeMixin):
    coords: Vec4
    """Cartesian coordinates (x, y, z, ct) [mm]"""

    @classmethod
    def make(
        cls,
        *,
        x: SFloat = 0.0,
        y: SFloat = 0.0,
        z: SFloat = 0.0,
        ct: SFloat | None = None,
        ctau: SFloat | None = None,
    ) -> Cartesian4:
        """Create Cartesian4 from individual components"""
        if ct is not None and ctau is not None:
            raise ValueError("Cannot specify both ct and ctau")
        elif ctau is not None:
            ct = jnp.sqrt(x**2 + y**2 + z**2 + ctau**2)
        elif ct is None:
            ct = 0.0
        return cls(coords=jnp.stack([x, y, z, ct], axis=-1))

    def to_cylindrical(self) -> Cylindric4:
        rho = jnp.hypot(self.x, self.y)
        phi = jnp.arctan2(self.y, self.x)
        return Cylindric4(coords=jnp.stack([rho, phi, self.z, self.ct], axis=-1))

    def to_cartesian3(self) -> Cartesian3:
        return Cartesian3(coords=self.coords[..., :3])

    def to_cylindric3(self) -> Cylindric3:
        return Cylindric3(coords=self.to_cylindrical().coords[..., :3])

    def __abs__(self) -> SFloat:
        metric = jnp.array([-1.0, -1.0, -1.0, 1.0])
        return jnp.sqrt(jnp.sum(metric * self.coords**2, axis=-1))

    def differential(self) -> Vec4:
        # TODO: correct for metric?
        return jnp.ones_like(self.coords)

    def volume_element(self) -> SFloat:
        return jnp.ones_like(self.x)


class Point[T: CoordinateChart](eqx.Module):
    """A point on a manifold, represented in a given coordinate chart"""

    x: T

    @overload
    def to_cartesian(self: Point[Cartesian3]) -> Point[Cartesian3]: ...
    @overload
    def to_cartesian(self: Point[Cartesian4]) -> Point[Cartesian4]: ...
    @overload
    def to_cartesian(self: Point[Cylindric3]) -> Point[Cartesian3]: ...
    @overload
    def to_cartesian(self: Point[Cylindric4]) -> Point[Cartesian4]: ...

    def to_cartesian(self) -> Point:
        return Point(x=self.x.to_cartesian())

    @overload
    def to_cylindrical(self: Point[Cartesian3]) -> Point[Cylindric3]: ...
    @overload
    def to_cylindrical(self: Point[Cylindric3]) -> Point[Cylindric3]: ...
    @overload
    def to_cylindrical(self: Point[Cylindric4]) -> Point[Cylindric4]: ...
    @overload
    def to_cylindrical(self: Point[Cartesian4]) -> Point[Cylindric4]: ...

    def to_cylindrical(self) -> Point:
        return Point(x=self.x.to_cylindrical())


class Tangent[T: CoordinateChart](eqx.Module):
    """A tangent vector on a manifold, represented in a given coordinate chart"""

    point: Point[T]
    dx: T

    def __abs__(self) -> SFloat:
        return abs(self.dx)

    def __add__(self, other: Tangent[T]) -> Tangent[T]:
        if self.point != other.point:
            raise ValueError("Cannot add tangent vectors at different points")
        return Tangent(
            point=self.point,
            dx=type(self.dx)(coords=self.dx.coords + other.dx.coords),
        )

    @overload
    def to_cartesian(self: Tangent[Cartesian3]) -> Tangent[Cartesian3]: ...
    @overload
    def to_cartesian(self: Tangent[Cartesian4]) -> Tangent[Cartesian4]: ...
    @overload
    def to_cartesian(self: Tangent[Cylindric3]) -> Tangent[Cartesian3]: ...
    @overload
    def to_cartesian(self: Tangent[Cylindric4]) -> Tangent[Cartesian4]: ...

    def to_cartesian(self) -> Tangent:
        tup: tuple[T, T] = jax.jvp(
            lambda v: v.to_cartesian(), (self.point.x,), (self.dx,)
        )
        x, dx = tup
        return Tangent(point=Point(x=x), dx=dx)

    @overload
    def to_cylindrical(
        self: Tangent[Cartesian3],
    ) -> Tangent[Cylindric3]: ...
    @overload
    def to_cylindrical(
        self: Tangent[Cylindric3],
    ) -> Tangent[Cylindric3]: ...
    @overload
    def to_cylindrical(
        self: Tangent[Cylindric4],
    ) -> Tangent[Cylindric4]: ...
    @overload
    def to_cylindrical(
        self: Tangent[Cartesian4],
    ) -> Tangent[Cylindric4]: ...

    def to_cylindrical(self) -> Tangent:
        tup: tuple[T, T] = jax.jvp(
            lambda v: v.to_cylindrical(), (self.point.x,), (self.dx,)
        )
        x, dx = tup
        return Tangent(point=Point(x=x), dx=dx)


class Cotangent[T: CoordinateChart](eqx.Module):
    """A cotangent vector on a manifold, represented in a given coordinate chart"""

    point: Point[T]
    dx: T


class GradientField[T: CoordinateChart](eqx.Module):
    """Gradient vector field of a scalar field"""

    field: Callable[[Point[T]], SFloat]

    # TODO: this should return Cotangent
    def __call__(self, point: Point[T]) -> Tangent[T]:
        grad: Point[T] = jax.grad(self.field)(point)
        value = type(point.x)(coords=grad.x.coords * point.x.differential())
        return Tangent(point=point, dx=value)


class DivergenceField[T: CoordinateChart](eqx.Module):
    """Divergence of a vector field"""

    field: Callable[[Point[T]], Tangent[T]]

    def __call__(self, point: Point[T]) -> SFloat:
        def func(p: Point[T]) -> VecN:
            return self.field(p).dx.coords * p.x.volume_element()

        jac: Point[T] = jax.jacobian(func)(point)
        return jnp.trace(jac.x.coords) / point.x.volume_element()


class Transform(eqx.Module):
    """A container for coordinate transformations

    This is a rigid transformation used to wrap fields defined in local
    coordinates to global coordinates. It is 4-dimensional to allow for
    time translations, and in principle also Lorentz boosts.

    Given a field f'(x', ...) defined in local coordinates x' = T x, this
    defines the transformed field via
        f(x, ...) = T^-1 f'(T x, ...)
    where
        T^-1 x = rotation @ x + translation
    (i.e. rotate and then translate the underlying object).
    """

    translation: Cartesian4
    """Translation vector [mm]"""
    rotation: Rot4Matrix
    """Rotation matrix (4x4)"""

    @classmethod
    def make_axis_angle(
        cls, axis: Cartesian3, angle: SFloat, translation: Cartesian4
    ) -> Self:
        """Create a Transform from an axis-angle rotation and translation"""

        # Rodrigues' rotation formula on a set of basis vectors
        across = jnp.array(
            [
                [0.0, axis.z, -axis.y],
                [-axis.z, 0.0, axis.x],
                [axis.y, -axis.x, 0.0],
            ]
        ) / abs(axis)
        rot_matrix = (
            jnp.eye(3)
            + jnp.sin(angle) * across
            + (1 - jnp.cos(angle)) * across @ across
        )
        rot4_matrix = jnp.eye(4).at[:3, :3].set(rot_matrix)
        return cls(translation=translation, rotation=rot4_matrix)

    # TODO maybe someday we can make this more generic

    def to_local(self, point: Point[Cartesian4]) -> Point[Cartesian4]:
        xyzg = point.x
        xyzl = jnp.linalg.inv(self.rotation) @ (xyzg.coords - self.translation.coords)
        return Point(x=Cartesian4(coords=xyzl))

    def to_global(self, point: Point[Cartesian4]) -> Point[Cartesian4]:
        xyzl = point.x
        xyzg = self.rotation @ xyzl.coords + self.translation.coords
        return Point(x=Cartesian4(coords=xyzg))

    def tangent_to_local(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        point_local = self.to_local(vec.point)
        dx_local = jnp.linalg.inv(self.rotation) @ vec.dx.coords
        return Tangent(point=point_local, dx=Cartesian4(coords=dx_local))

    def tangent_to_global(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        point_global = self.to_global(vec.point)
        dx_global = self.rotation @ vec.dx.coords
        return Tangent(point=point_global, dx=Cartesian4(coords=dx_global))

    # TODO: implement for Cotangent?


class TransformOneForm(eqx.Module):
    """A container to transform 1-forms (i.e. gradients)"""

    transform: Transform
    field: Callable[[Point[Cartesian4]], Tangent[Cartesian4]]
    """The field in local coordinates (i.e. before transformation)"""

    # TODO: this should return Cotangent
    def __call__(self, point: Point[Cartesian4]) -> Tangent[Cartesian4]:
        in_local = self.transform.to_local(point)
        out_local = self.field(in_local)
        return self.transform.tangent_to_global(out_local)
