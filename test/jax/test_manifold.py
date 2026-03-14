import jax.numpy as jnp
import pytest

from beamline.jax.manifold import (
    XYZT,
    RhoPhiZT,
    Tangent,
    d,
    lie_bracket,
    pullback,
    pushforward,
)
from beamline.jax.types import SFloat


def test_lower():
    p = XYZT.make(x=1.0, y=2.0, z=3.0, t=4.0)
    t = Tangent.make(p=p, t=jnp.array([1.0, 2.0, 3.0, 4.0]))
    ct = XYZT.lower(t)
    expected = jnp.array([-1.0, -2.0, -3.0, 4.0])
    assert ct.ct == pytest.approx(expected)


def test_pushpull():
    p = XYZT.make(x=2.0, y=0.0, z=3.0, t=4.0)
    t = Tangent.make(p=p, t=jnp.array([0.0, 1.0, 0.0, 4.0]))
    tnew = pushforward(XYZT.to_rhophizt, t)
    expected = jnp.array([0.0, 0.5, 0.0, 4.0])
    assert tnew.t == pytest.approx(expected)
    assert tnew.p.metric(tnew, tnew) == pytest.approx(p.metric(t, t))

    ct = pullback(RhoPhiZT.to_xyzt, tnew.p)(p.lower(t))
    assert ct(tnew) == pytest.approx(p.metric(t, t))


def test_d():
    p = XYZT.make(x=1.0, y=2.0, z=3.0, t=4.0)

    def func(p: XYZT) -> SFloat:
        return p.x**2 + p.y**2 + p.z**2 + p.t**2

    w = d(func)(p)
    assert w.ct == pytest.approx(jnp.array([2.0, 4.0, 6.0, 8.0]))


def test_lie():
    def drho(p: RhoPhiZT) -> Tangent[RhoPhiZT]:
        return Tangent.make(p=p, t=jnp.array([1.0, 0.0, 0.0, 0.0]))

    def dphi(p: RhoPhiZT) -> Tangent[RhoPhiZT]:
        return Tangent.make(p=p, t=jnp.array([0.0, 1.0, 0.0, 0.0]))

    thing = lie_bracket(drho, dphi)
    p = RhoPhiZT.make(rho=1.0, phi=2.0, z=3.0, t=4.0)
    assert thing(p).t == pytest.approx(jnp.array([0.0, 0.0, 0.0, 0.0]))
