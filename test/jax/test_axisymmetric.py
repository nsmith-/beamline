from itertools import product

import hepunits as u
import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian4
from beamline.jax.rfcavity.pillbox import PillboxCavity

M0_MODES = [
    (n, p, mode)
    for n, p, mode in product(range(1, 3), range(3), ("TE", "TM"))
    if not (mode == "TE" and p == 0)
]


def _cavity(n: int, p: int, mode: str) -> PillboxCavity:
    return PillboxCavity(
        length=1 * u.m,
        frequency=704 * u.MHz,
        E0=30 * u.megavolt / u.m,
        mode=mode,
        m=0,
        n=n,
        p=p,
        phase=0.3,
    )


@pytest.mark.parametrize(("n", "p", "mode"), M0_MODES)
def test_pillbox_m0_matches_cylindric(n: int, p: int, mode: str):
    """The axisymmetric path reproduces the cylindrical one away from the axis"""
    cavity = _cavity(n, p, mode)
    R = float(cavity.radius)
    for x, y, z, ct in [
        (0.3 * R, 0.2 * R, 120.0, 50.0),
        (-0.5 * R, 0.4 * R, -300.0, 400.0),
        (0.01 * R, -0.02 * R, 10.0, 900.0),
        (1e-3, 2e-3, 30.0, 200.0),
    ]:
        point = Cartesian4.make(x=x, y=y, z=z, ct=ct)
        E, B = cavity.field_strength(point)
        Ec, Bc = cavity._cylindric_field(point.to_cylindric3(), point.ct)
        Ec, Bc = Ec.to_cartesian(), Bc.to_cartesian()
        for got, ref in ((E, Ec), (B, Bc)):
            scale = jnp.max(jnp.abs(ref.t.coords))
            assert jnp.allclose(got.t.coords, ref.t.coords, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize(("n", "p", "mode"), M0_MODES)
def test_pillbox_m0_on_axis(n: int, p: int, mode: str):
    """On axis the field and its Jacobian are finite; transverse components vanish"""
    cavity = _cavity(n, p, mode)

    def EB(xyzt):
        E, B = cavity.field_strength(
            Cartesian4.make(x=xyzt[0], y=xyzt[1], z=xyzt[2], ct=xyzt[3])
        )
        return jnp.concatenate([E.t.coords, B.t.coords])

    for z in (0.0, 123.0, -400.0):
        q = jnp.array([0.0, 0.0, z, 250.0])
        F = EB(q)
        assert jnp.all(jnp.isfinite(F))
        assert jnp.all(F[jnp.array([0, 1, 3, 4])] == 0.0)
        J = jax.jacfwd(EB)(q)
        assert jnp.all(jnp.isfinite(J))
        # div E = div B = 0
        for i in (0, 3):
            scale = jnp.abs(J[i : i + 3, :3]).max()
            assert float(jnp.trace(J[i : i + 3, :3])) == pytest.approx(
                0.0, abs=1e-12 * float(scale) + 1e-300
            )
