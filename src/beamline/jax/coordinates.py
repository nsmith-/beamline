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
    def to_cylindric(self) -> Cylindric:
        """Convert to Cylindrical coordinates"""

    @abstractmethod
    def __abs__(self) -> SFloat:
        """Norm of the vector"""

    @abstractmethod
    def lame_coefficients(self) -> T:
        r"""Lengths of the covariant basis vectors $e_i$

        See https://en.wikipedia.org/wiki/Orthogonal_coordinates#Basis_vectors
        """

    @abstractmethod
    def volume_element(self) -> SFloat:
        """Volume element in this coordinate chart"""


class Cylindric[T: VecN](CoordinateChart[T]):
    def to_cylindric(self) -> Self:
        return self


class Cartesian[T: VecN](CoordinateChart[T]):
    def to_cartesian(self) -> Self:
        return self


def delta_phi(phi1: SFloat, phi2: SFloat) -> SFloat:
    """Compute (phi1-phi2) wrapped to [-pi, pi)"""
    dphi = phi1 - phi2
    return (dphi + jnp.pi) % (2 * jnp.pi) - jnp.pi


class XYMixin:
    coords: Vec3 | Vec4

    @property
    def x(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def y(self) -> SFloat:
        return self.coords[..., 1]

    @property
    def rho(self) -> SFloat:
        return jnp.hypot(self.x, self.y)

    @property
    def phi(self) -> SFloat:
        return jnp.atan2(self.y, self.x)

    def delta_phi(self, other: Self) -> SFloat:
        return delta_phi(self.phi, other.phi)


class PolarMixin:
    coords: Vec3 | Vec4

    @property
    def x(self) -> SFloat:
        return self.rho * jnp.cos(self.phi)

    @property
    def y(self) -> SFloat:
        return self.rho * jnp.sin(self.phi)

    @property
    def rho(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def phi(self) -> SFloat:
        return self.coords[..., 1]

    def delta_phi(self, other: Self) -> SFloat:
        return delta_phi(self.phi, other.phi)


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


class Cylindric3(Cylindric[Vec3], PolarMixin, ZMixin):
    coords: Vec3
    """Cylindrical coordinates (rho, phi, z) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0
    ) -> Cylindric3:
        """Create Cylindric3 from individual components"""
        return cls(coords=jnp.stack([rho, phi, z], axis=-1))

    def to_cartesian(self) -> Cartesian3:
        return Cartesian3.make(x=self.x, y=self.y, z=self.z)

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.rho**2 + self.z**2)

    def lame_coefficients(self) -> Vec3:
        # drho, rho dphi, dz
        return jnp.ones_like(self.coords).at[..., 1].set(self.rho)

    def volume_element(self) -> SFloat:
        return self.rho

    # TODO: implement add, mul, dot, cross etc. (or maybe we should just convert to Cartesian for that?)


class Cartesian3(Cartesian[Vec3], XYMixin, ZMixin):
    coords: Vec3
    """Cartesian coordinates (x, y, z) [mm]"""

    @classmethod
    def make(cls, *, x: SFloat = 0.0, y: SFloat = 0.0, z: SFloat = 0.0) -> Cartesian3:
        """Create Cartesian3 from individual components"""
        return cls(coords=jnp.stack([x, y, z], axis=-1))

    def to_cylindric(self) -> Cylindric3:
        return Cylindric3.make(rho=self.rho, phi=self.phi, z=self.z)

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)

    def lame_coefficients(self) -> Vec3:
        return jnp.ones_like(self.coords)

    def volume_element(self) -> SFloat:
        return jnp.ones_like(self.x)

    def __mul__(self, scalar: SFloat) -> Self:
        return type(self)(coords=self.coords * scalar)

    def __rmul__(self, scalar: SFloat) -> Self:
        return self * scalar

    def __add__(self, other: Cartesian3) -> Cartesian3:
        return Cartesian3(coords=self.coords + other.coords)

    def __sub__(self, other: Cartesian3) -> Cartesian3:
        return Cartesian3(coords=self.coords - other.coords)

    # TODO: should these only be defined on Tangent[Cartesian3]?

    def dot(self, other: Cartesian3) -> SFloat:
        return jnp.sum(self.coords * other.coords, axis=-1)

    def cross(self, other: Cartesian3) -> Cartesian3:
        return Cartesian3.make(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )


class Cylindric4(Cylindric[Vec4], PolarMixin, ZMixin, TimeMixin):
    coords: Vec4
    """Cylindrical coordinates (rho, phi, z, ct) [mm]"""

    @classmethod
    def make(
        cls, *, rho: SFloat = 0.0, phi: SFloat = 0.0, z: SFloat = 0.0, ct: SFloat = 0.0
    ) -> Cylindric4:
        """Create Cylindric4 from individual components"""
        return cls(coords=jnp.stack([rho, phi, z, ct], axis=-1))

    def to_cartesian(self) -> Cartesian4:
        return Cartesian4.make(x=self.x, y=self.y, z=self.z, ct=self.ct)

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.ct**2 - self.rho**2 - self.z**2)

    def lame_coefficients(self) -> Vec4:
        return jnp.ones_like(self.coords).at[..., 1].set(self.rho)

    def volume_element(self) -> SFloat:
        return self.rho


class Cartesian4(Cartesian[Vec4], XYMixin, ZMixin, TimeMixin):
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

    def to_cylindric(self) -> Cylindric4:
        return Cylindric4.make(rho=self.rho, phi=self.phi, z=self.z, ct=self.ct)

    def to_cartesian3(self) -> Cartesian3:
        return Cartesian3(coords=self.coords[..., :3])

    def to_cylindric3(self) -> Cylindric3:
        return Cylindric3(coords=self.to_cylindric().coords[..., :3])

    def __abs__(self) -> SFloat:
        return jnp.sqrt(self.dot(self))

    def lame_coefficients(self) -> Vec4:
        return jnp.ones_like(self.coords)

    def volume_element(self) -> SFloat:
        return jnp.ones_like(self.x)

    def __mul__(self, scalar: SFloat) -> Self:
        return type(self)(coords=self.coords * scalar)

    def __rmul__(self, scalar: SFloat) -> Self:
        return self * scalar

    def __add__(self, other: Cartesian4) -> Cartesian4:
        return Cartesian4(coords=self.coords + other.coords)

    def __sub__(self, other: Cartesian4) -> Cartesian4:
        return Cartesian4(coords=self.coords - other.coords)

    def dot(self, other: Cartesian4) -> SFloat:
        # Flat space for now :)
        metric = jnp.array([-1.0, -1.0, -1.0, 1.0])
        return jnp.sum(metric * (self.coords * other.coords), axis=-1)


class Tangent[T: CoordinateChart](eqx.Module):
    """A tangent vector on a manifold, represented in a given coordinate chart

    The tangent basis vectors are **normalized** so that the coordinates of the
    tangent vector are the physical components of the vector and always have
    uniform units (e.g. mm for spatial components).
    """

    p: T
    """Point """
    t: T
    """Tangent vector at p"""

    def __abs__(self) -> SFloat:
        return abs(self.t)

    # In any coordinate chart, the tangents are a vector space

    def __add__(self, other: Tangent[T]) -> Tangent[T]:
        if self.p != other.p:
            raise ValueError("Cannot add tangent vectors at different points")
        return Tangent(
            p=self.p,
            t=type(self.t)(coords=self.t.coords + other.t.coords),
        )

    def __sub__(self, other: Tangent[T]) -> Tangent[T]:
        if self.p != other.p:
            raise ValueError("Cannot subtract tangent vectors at different points")
        return Tangent(
            p=self.p,
            t=type(self.t)(coords=self.t.coords - other.t.coords),
        )

    def __mul__(self, scalar: SFloat) -> Tangent[T]:
        return Tangent(
            p=self.p,
            t=type(self.t)(coords=self.t.coords * scalar),
        )

    def __rmul__(self, scalar: SFloat) -> Tangent[T]:
        return self * scalar

    def __truediv__(self, scalar: SFloat) -> Tangent[T]:
        return Tangent(
            p=self.p,
            t=type(self.t)(coords=self.t.coords / scalar),
        )

    def __rtruediv__(self, scalar: SFloat) -> Tangent[T]:
        return self / scalar

    def _change_basis[U: CoordinateChart](self, func: Callable[[T], U]) -> Tangent[U]:
        """Change the coordinate chart of the tangent vector using the given function

        This is a bit delicate because we want to maintain unit normalization of each
        basis vector.

        Args:
            func: Function that changes basis of the manifold point (e.g. to_cartesian or to_cylindric)
        """
        tscaled = type(self.t)(coords=self.t.coords / self.p.lame_coefficients())
        p_out, t_out = jax.jvp(func, (self.p,), (tscaled,))
        t_out_norm = type(t_out)(coords=t_out.coords * p_out.lame_coefficients())
        return Tangent(p=p_out, t=t_out_norm)

    @overload
    def to_cartesian(self: Tangent[Cylindric3]) -> Tangent[Cartesian3]: ...
    @overload
    def to_cartesian(self: Tangent[Cylindric4]) -> Tangent[Cartesian4]: ...

    def to_cartesian(self) -> Tangent:
        def func(v: T):
            return v.to_cartesian()

        return self._change_basis(func)

    @overload
    def to_cylindric(self: Tangent[Cartesian3]) -> Tangent[Cylindric3]: ...
    @overload
    def to_cylindric(self: Tangent[Cartesian4]) -> Tangent[Cylindric4]: ...

    def to_cylindric(self) -> Tangent:
        def func(v: T) -> Cylindric:
            return v.to_cylindric()

        return self._change_basis(func)


class Cotangent[T: CoordinateChart](eqx.Module):
    """TODO"""

    p: T
    """Point"""
    t_co: T
    """Cotangent vector at p"""


class GradientField[T: CoordinateChart](eqx.Module):
    """Gradient vector field of a scalar field

    This probably only works for orthogonal coordinates at the moment
    Generalize with: https://en.wikipedia.org/wiki/Gradient#Riemannian_manifolds

    Note: jax.grad will return the gradient with respect to the coordinates, which
    are the covariant basis vectors. We have to manually divide by the Lamé coefficients
    to have normalized tangents.
    https://en.wikipedia.org/wiki/Orthogonal_coordinates#Contravariant_basis
    """

    field: Callable[[T], SFloat]

    def __call__(self, point: T) -> Tangent[T]:
        grad: T = jax.grad(self.field)(point)
        val = type(point)(coords=grad.coords / point.lame_coefficients())
        return Tangent(p=point, t=val)


class DivergenceField[T: CoordinateChart](eqx.Module):
    """Divergence of a vector field

    Note: the general form requires un-normalized basis vectors
    https://en.wikipedia.org/wiki/Divergence#General_coordinates
    """

    field: Callable[[T], Tangent[T]]

    def __call__(self, point: T) -> SFloat:
        def func(p: T) -> VecN:
            return self.field(p).t.coords * p.volume_element() / p.lame_coefficients()

        jac: T = jax.jacobian(func)(point)
        return jnp.trace(jac.coords) / point.volume_element()


class Transform(eqx.Module):
    """A container for coordinate transformations

    This is a rigid transformation used to wrap fields defined in local
    coordinates to global coordinates. It is 4-dimensional to allow for
    time translations, and in principle also Lorentz boosts, though for
    the latter, a proper separation between covariant and contravariant
    tensors is required.

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
                [0.0, -axis.z, axis.y],
                [axis.z, 0.0, -axis.x],
                [-axis.y, axis.x, 0.0],
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
    @overload
    def to_local(self, point: Cartesian4) -> Cartesian4: ...
    @overload
    def to_local(self, point: Cartesian3) -> Cartesian3: ...

    def to_local(self, point: Cartesian) -> Cartesian:
        if isinstance(point, Cartesian4):
            xyzl = jnp.linalg.inv(self.rotation) @ (
                point.coords - self.translation.coords
            )
        elif isinstance(point, Cartesian3):
            xyzl = jnp.linalg.inv(self.rotation)[:3, :3] @ (
                point.coords - self.translation.coords[:3]
            )
        else:
            raise ValueError("Unsupported coordinate dimension")
        return type(point)(coords=xyzl)

    @overload
    def to_global(self, point: Cartesian4) -> Cartesian4: ...
    @overload
    def to_global(self, point: Cartesian3) -> Cartesian3: ...

    def to_global(self, point: Cartesian) -> Cartesian:
        if isinstance(point, Cartesian4):
            xyzg = self.rotation @ point.coords + self.translation.coords
        elif isinstance(point, Cartesian3):
            xyzg = self.rotation[:3, :3] @ point.coords + self.translation.coords[:3]
        else:
            raise ValueError("Unsupported coordinate dimension")
        return type(point)(coords=xyzg)

    def tangent_to_local[T: Cartesian](self, vec: Tangent[T]) -> Tangent[T]:
        p, t = jax.jvp(self.to_local, (vec.p,), (vec.t,))
        return Tangent(p=p, t=t)

    def tangent_to_global[T: Cartesian](self, vec: Tangent[T]) -> Tangent[T]:
        p, t = jax.jvp(self.to_global, (vec.p,), (vec.t,))
        return Tangent(p=p, t=t)

    # TODO: implement for Cotangent?


class TransformOneForm[T: Cartesian3 | Cartesian4](eqx.Module):
    """A container to transform 1-forms (i.e. gradients)"""

    transform: Transform
    field: Callable[[T], Tangent[T]]
    """The field in local coordinates (i.e. before transformation)"""

    # TODO: this should return Cotangent
    def __call__(self, point: T) -> Tangent[T]:
        in_local = self.transform.to_local(point)
        out_local = self.field(in_local)
        return self.transform.tangent_to_global(out_local)
