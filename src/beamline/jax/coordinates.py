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

from beamline.jax.types import SFloat, Vec3, Vec4, VecN


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


class TangentVector[T: CoordinateChart](eqx.Module):
    """A tangent vector on a manifold, represented in a given coordinate chart"""

    point: Point[T]
    dx: T

    def __abs__(self) -> SFloat:
        return abs(self.dx)

    def __add__(self, other: TangentVector[T]) -> TangentVector[T]:
        if self.point != other.point:
            raise ValueError("Cannot add tangent vectors at different points")
        return TangentVector(
            point=self.point,
            dx=type(self.dx)(coords=self.dx.coords + other.dx.coords),
        )

    @overload
    def to_cartesian(self: TangentVector[Cartesian3]) -> TangentVector[Cartesian3]: ...
    @overload
    def to_cartesian(self: TangentVector[Cartesian4]) -> TangentVector[Cartesian4]: ...
    @overload
    def to_cartesian(self: TangentVector[Cylindric3]) -> TangentVector[Cartesian3]: ...
    @overload
    def to_cartesian(self: TangentVector[Cylindric4]) -> TangentVector[Cartesian4]: ...

    def to_cartesian(self) -> TangentVector:
        tup: tuple[T, T] = jax.jvp(
            lambda v: v.to_cartesian(), (self.point.x,), (self.dx,)
        )
        p, t = tup
        return TangentVector(point=Point(x=p), dx=t)

    @overload
    def to_cylindrical(
        self: TangentVector[Cartesian3],
    ) -> TangentVector[Cylindric3]: ...
    @overload
    def to_cylindrical(
        self: TangentVector[Cylindric3],
    ) -> TangentVector[Cylindric3]: ...
    @overload
    def to_cylindrical(
        self: TangentVector[Cylindric4],
    ) -> TangentVector[Cylindric4]: ...
    @overload
    def to_cylindrical(
        self: TangentVector[Cartesian4],
    ) -> TangentVector[Cylindric4]: ...

    def to_cylindrical(self) -> TangentVector:
        tup: tuple[T, T] = jax.jvp(
            lambda v: v.to_cylindrical(), (self.point.x,), (self.dx,)
        )
        x, dx = tup
        return TangentVector(point=Point(x=x), dx=dx)


class GradientField[T: CoordinateChart](eqx.Module):
    """Gradient vector field of a scalar field"""

    field: Callable[[Point[T]], SFloat]

    def __call__(self, point: Point[T]) -> TangentVector[T]:
        grad: Point[T] = jax.grad(self.field)(point)
        value = type(point.x)(coords=grad.x.coords * point.x.differential())
        return TangentVector(point=point, dx=value)


class DivergenceField[T: CoordinateChart](eqx.Module):
    """Divergence of a vector field"""

    field: Callable[[Point[T]], TangentVector[T]]

    def __call__(self, point: Point[T]) -> SFloat:
        def func(p: Point[T]) -> VecN:
            return self.field(p).dx.coords * p.x.volume_element()

        jac: Point[T] = jax.jacobian(func)(point)
        return jnp.trace(jac.x.coords) / point.x.volume_element()
