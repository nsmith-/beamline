"""Attempt 2 at coordinates"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, Self

import jax
import jax.numpy as jnp

from beamline.jax.types import SFloat


class Point(Protocol):
    """A coordinate chart point on a manifold"""

    coords: jax.Array
    """Coordinates

    i.e. a local representation of the manifold point
    May be different from internal representation if embedded/immersed
    in a higher-dimensional space
    """

    def __init__(self, coords: jax.Array, aux: Any = None) -> None:
        """Construct

        Always need to implement, for wrapping in pushforward/pullback.
        User interface should be through classmethod make() instead of directly calling __init__.

        Args:
            coords: Coordinates of the point
            aux: Non-differentiable auxiliary data for the point, e.g. chart id, if necessary
        """
        ...

    def dim(self) -> int:
        """Dimension

        Must be same as coords.shape[-1]
        """
        ...

    def equal(self, other: Self) -> bool:
        """Equality of points

        This may be different from equality of coordinates if there are multiple charts
        """
        ...

    # TODO: exponential map? (if connection is defined)

    # TODO: auto-change chart when necessary (need to add chart id to data members)


class MetricPoint(Point, Protocol):
    """Embue point on manifold with a metric"""

    @classmethod
    def metric(cls, a: Tangent[Self], b: Tangent[Self]) -> SFloat:
        """The metric evaluated on two tangent vectors at a point"""
        ...

    @classmethod
    def lower(cls, t: Tangent[Self]) -> Cotangent[Self]:
        """Lower a tangent vector to a cotangent vector using the metric"""
        ...


class MetricMixin(MetricPoint):
    """Mixin class to implement lower() using the given metric()"""

    @classmethod
    def lower(cls, t: Tangent[Self]) -> Cotangent[Self]:
        p = t.p
        ct = jnp.stack(
            [p.metric(t, type(t).make(p=p, t=tbasis)) for tbasis in jnp.eye(p.dim())],
            axis=-1,
        )
        return Cotangent.make(p=p, ct=ct)

    # TODO: raise


class BarePoint:
    """Point that has one chart and internal representation same as coordinates

    Equality is just coordinate equality
    """

    def __init__(self, coords: jax.Array, aux=None) -> None:
        self.coords = coords

    def dim(self) -> int:
        return self.coords.shape[-1]

    def equal(self, other: Self) -> bool:
        # when tracing, should be the same object (usually)
        # when directly evaluating, coordinates will exist and can be compared
        return (self is other) or bool(jnp.allclose(self.coords, other.coords))


class Tangent[T: Point]:
    """A tangent vector to a manifold at a given point, with coordinates of type T

    It itself is a manifold point on TM
    """

    def __init__(self, coords: jax.Array, aux: type[T] | None = None) -> None:
        if aux is None:
            raise ValueError("Always construct with make() to ensure aux is set")
        self.coords = coords
        self.p_cls = aux

    def dim(self) -> int:
        return self.coords.shape[-1]

    def equal(self, other: Self) -> bool:
        return (self is other) or bool(jnp.allclose(self.coords, other.coords))

    @classmethod
    def make(cls, /, p: T, t: jax.Array) -> Self:
        assert p.coords.shape[-1] == t.shape[-1]
        coords = jnp.concatenate([p.coords, t], axis=-1)
        return cls(coords=coords, aux=type(p))

    @property
    def p(self) -> T:
        dim = self.dim() // 2
        return self.p_cls(coords=self.coords[..., :dim])

    @property
    def t(self) -> jax.Array:
        dim = self.dim() // 2
        return self.coords[..., dim:]

    def __call__(self, other: Cotangent[T]) -> SFloat:
        return other.ct @ self.t


class Cotangent[T: Point]:
    """A cotangent vector to a manifold at a given point, with coordinates of type T

    It itself is a manifold point on T*M
    """

    def __init__(self, coords: jax.Array, aux: type[T] | None = None) -> None:
        if aux is None:
            raise ValueError("Always construct with make() to ensure aux is set")
        self.coords = coords
        self.p_cls = aux

    def dim(self) -> int:
        return self.coords.shape[-1]

    def equal(self, other: Self) -> bool:
        return (self is other) or bool(jnp.allclose(self.coords, other.coords))

    @classmethod
    def make(cls, /, p: T, ct: jax.Array) -> Self:
        assert p.coords.shape[-1] == ct.shape[-1]
        coords = jnp.concatenate([p.coords, ct], axis=-1)
        return cls(coords=coords, aux=type(p))

    @property
    def p(self) -> T:
        dim = self.dim() // 2
        return self.p_cls(coords=self.coords[..., :dim])

    @property
    def ct(self) -> jax.Array:
        dim = self.dim() // 2
        return self.coords[..., dim:]

    def __call__(self, other: Tangent[T]) -> SFloat:
        return self.ct @ other.t


def _func_wrap[M: Point, N: Point](
    m_cls: type[M], func: Callable[[M], N], m_coords: jax.Array
) -> tuple[jax.Array, type[N]]:
    """Rather than deal with pytrees, we use the coordinates and reconstruct types"""
    m_p = m_cls(coords=m_coords)
    n_p = func(m_p)
    return n_p.coords, type(n_p)


def pushforward[M: Point, N: Point](
    func: Callable[[M], N], v: Tangent[M]
) -> Tangent[N]:
    """Pushforward of a tangent vector v by a function func: M -> N"""
    func_coords = partial(_func_wrap, type(v.p), func)
    n_coords, n_t, n_cls = jax.jvp(func_coords, (v.p.coords,), (v.t,), has_aux=True)
    return Tangent.make(p=n_cls(coords=n_coords), t=n_t)


def pullback[M: Point, N: Point](
    func: Callable[[M], N], m_p: M
) -> Callable[[Cotangent[N]], Cotangent[M]]:
    """Pullback from T*N to T*M at a point p by a function func: M -> N"""
    func_coords = partial(_func_wrap, type(m_p), func)
    n_coords, vjp, n_cls = jax.vjp(func_coords, m_p.coords, has_aux=True)
    n_p = n_cls(coords=n_coords)

    def out(w: Cotangent[N]) -> Cotangent[M]:
        if not n_p.equal(w.p):
            raise ValueError("Points must be the same for pullback evaluation")
        (ct,) = vjp(w.ct)
        return Cotangent.make(p=m_p, ct=ct)

    return out


ZERO = jnp.zeros(())


class RealLine(BarePoint):
    """Real line (1D manifold)"""

    @classmethod
    def make(cls, /, x: SFloat = ZERO) -> Self:
        return cls(coords=jnp.array([x]))

    @property
    def x(self) -> SFloat:
        return self.coords[..., 0]

    @classmethod
    def metric(cls, a: Tangent[Self], b: Tangent[Self]) -> SFloat:
        return a.t * b.t


def _real_as_manifold[M: Point](func: Callable[[M], SFloat], p: M) -> RealLine:
    return RealLine.make(x=func(p))


def d[M: Point](func: Callable[[M], SFloat]) -> Callable[[M], Cotangent[M]]:
    """Differential

    Exterior derivative / Lie derivative / covariant derivative, all the same for
    functions returning scalars. Returns a 1-form

    TODO: make this exterior derivative for arbitrary forms, not just 0-forms
    """

    def out(m_p: M) -> Cotangent[M]:
        w = Cotangent.make(p=RealLine.make(x=func(m_p)), ct=jnp.array([1.0]))
        func_real = partial(_real_as_manifold, func)
        return pullback(func_real, m_p)(w)

    return out


def exterior_derivative[*Ts, M: Point](
    form: Callable[[*Ts], SFloat], m_p: M
) -> Callable[[Tangent[M], *Ts], SFloat]:
    """Exterior derivative of a form"""
    raise NotImplementedError("TODO")


def lie_bracket[M: MetricPoint](
    X: Callable[[M], Tangent[M]], Y: Callable[[M], Tangent[M]]
) -> Callable[[M], Tangent[M]]:
    """Lie bracket of two vector fields at a point"""

    def out(p: M) -> Tangent[M]:
        dXY = pushforward(Y, X(p))
        dYX = pushforward(X, Y(p))
        return Tangent.make(
            p=p,
            # latter half of TTM has the change in the tangent vector
            t=(dXY.t - dYX.t)[..., p.dim() :],
        )

    return out


class XYZT(BarePoint, MetricMixin):
    @classmethod
    def make(
        cls,
        /,
        x: SFloat = ZERO,
        y: SFloat = ZERO,
        z: SFloat = ZERO,
        t: SFloat = ZERO,
    ) -> Self:
        return cls(coords=jnp.stack([x, y, z, t], axis=-1))

    @classmethod
    def metric(cls, a: Tangent[XYZT], b: Tangent[XYZT]) -> SFloat:
        if not a.p.equal(b.p):
            raise ValueError("Points must be the same for metric evaluation")
        return jnp.sum(a.t * b.t * jnp.array([-1, -1, -1, 1]), axis=-1)

    @property
    def x(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def y(self) -> SFloat:
        return self.coords[..., 1]

    @property
    def z(self) -> SFloat:
        return self.coords[..., 2]

    @property
    def t(self) -> SFloat:
        return self.coords[..., 3]

    def to_rhophizt(self) -> RhoPhiZT:
        rho = jnp.hypot(self.x, self.y)
        phi = jnp.arctan2(self.y, self.x)
        return RhoPhiZT.make(rho=rho, phi=phi, z=self.z, t=self.t)


class RhoPhiZT(BarePoint, MetricMixin):
    @classmethod
    def make(
        cls,
        /,
        rho: SFloat = ZERO,
        phi: SFloat = ZERO,
        z: SFloat = ZERO,
        t: SFloat = ZERO,
    ) -> Self:
        return cls(coords=jnp.stack([rho, phi, z, t], axis=-1))

    @classmethod
    def metric(cls, a: Tangent[RhoPhiZT], b: Tangent[RhoPhiZT]) -> SFloat:
        if not a.p.equal(b.p):
            raise ValueError("Points must be the same for metric evaluation")
        rho = a.p.coords[0]
        return jnp.sum(a.t * b.t * jnp.array([-1, -(rho**2), -1, 1]), axis=-1)

    @property
    def rho(self) -> SFloat:
        return self.coords[..., 0]

    @property
    def phi(self) -> SFloat:
        return self.coords[..., 1]

    @property
    def z(self) -> SFloat:
        return self.coords[..., 2]

    @property
    def t(self) -> SFloat:
        return self.coords[..., 3]

    def to_xyzt(self) -> XYZT:
        x = self.rho * jnp.cos(self.phi)
        y = self.rho * jnp.sin(self.phi)
        return XYZT.make(x=x, y=y, z=self.z, t=self.t)
