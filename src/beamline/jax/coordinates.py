"""Coordinate systems implemented in equinox"""

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


class Cylindric3(Cylindric):
    coords: Vec3
    """Cylindrical coordinates (rho, phi, z) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0
    ) -> Cylindric3:
        """Create Cylindric3 from individual components"""
        return cls(coords=jnp.array([rho, phi, z]))

    def to_cartesian(self) -> Cartesian3:
        rho, phi, z = self.coords
        x = rho * jnp.cos(phi)
        y = rho * jnp.sin(phi)
        return Cartesian3(coords=jnp.array([x, y, z]))

    def __abs__(self) -> SFloat:
        rho, _, z = self.coords
        return jnp.sqrt(rho**2 + z**2)

    def differential(self) -> Vec3:
        return jnp.array([1.0, self.coords[0], 1.0])

    def volume_element(self) -> SFloat:
        return self.coords[0]


class Cartesian3(Cartesian):
    coords: Vec3
    """Cartesian coordinates (x, y, z) [mm]"""

    @classmethod
    def make(cls, *, x: SFloat = 0.0, y: SFloat = 0.0, z: SFloat = 0.0) -> Cartesian3:
        """Create Cartesian3 from individual components"""
        return cls(coords=jnp.array([x, y, z]))

    def to_cylindrical(self) -> Cylindric3:
        x, y, z = self.coords
        rho = jnp.hypot(x, y)
        phi = jnp.arctan2(y, x)
        return Cylindric3(coords=jnp.array([rho, phi, z]))

    def __abs__(self) -> SFloat:
        x, y, z = self.coords
        return jnp.sqrt(x**2 + y**2 + z**2)

    def differential(self) -> Vec3:
        return jnp.ones(3)

    def volume_element(self) -> SFloat:
        return 1.0


class Cylindric4(Cylindric):
    coords: Vec4
    """Cylindrical coordinates (rho, phi, z, ct) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0, ct: SFloat = 0.0
    ) -> Cylindric4:
        """Create Cylindric4 from individual components"""
        return cls(coords=jnp.array([rho, phi, z, ct]))

    def to_cartesian(self) -> Cartesian4:
        rho, phi, z, ct = self.coords
        x = rho * jnp.cos(phi)
        y = rho * jnp.sin(phi)
        return Cartesian4(coords=jnp.array([x, y, z, ct]))

    def __abs__(self) -> SFloat:
        rho, _, z, ct = self.coords
        return jnp.sqrt(ct**2 - rho**2 - z**2)

    def differential(self) -> Vec4:
        # TODO: correct for metric?
        return jnp.array([1.0, self.coords[0], 1.0, 1.0])

    def volume_element(self) -> SFloat:
        return self.coords[0]


class Cartesian4(Cartesian):
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
        return cls(coords=jnp.array([x, y, z, ct]))

    def to_cylindrical(self) -> Cylindric4:
        x, y, z, ct = self.coords
        rho = jnp.hypot(x, y)
        phi = jnp.arctan2(y, x)
        return Cylindric4(coords=jnp.array([rho, phi, z, ct]))

    def to_cartesian3(self) -> Cartesian3:
        return Cartesian3(coords=self.coords[:3])

    def __abs__(self) -> SFloat:
        metric = jnp.array([-1.0, -1.0, -1.0, 1.0])
        return jnp.sqrt(jnp.sum(metric * self.coords**2))

    def differential(self) -> Vec4:
        # TODO: correct for metric?
        return jnp.ones(4)

    def volume_element(self) -> SFloat:
        return 1.0


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
