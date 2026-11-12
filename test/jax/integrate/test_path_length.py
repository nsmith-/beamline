"""Tests for ``MuonStateDs``: propagation vs path length, with proper time"""

import hepunits as u
import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDs

SAVE_S = jnp.linspace(0.0, 1000.0 * u.mm, 11)


def _solve(field: SimpleEMField, p3: Cartesian3) -> MuonStateDs:
    start = MuonStateDs.make(position=Cartesian4.make(), momentum=p3, q=1)
    ys, _ = diffrax_solve(field, start, SAVE_S, rtol=1e-9, atol=1e-9)
    return ys


def test_make_ctau():
    start = MuonStateDs.make(
        position=Cartesian4.make(x=jnp.array([0.0, 1.0]) * u.mm),
        momentum=Cartesian3.make(z=100.0 * u.MeV),
        q=1,
        ctau=5.0 * u.mm,
    )
    assert start.ctau.shape == (2,)
    assert jnp.all(start.ctau == 5.0 * u.mm)


def test_free_field():
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    p3 = Cartesian3.make(x=30.0 * u.MeV, y=-40.0 * u.MeV, z=120.0 * u.MeV)
    ys = _solve(field, p3)
    pmag = abs(p3)
    energy = ys.kin.t.ct[0]
    direction = p3.coords / pmag
    assert ys.kin.p.coords[:, :3] == pytest.approx(SAVE_S[:, None] * direction)
    assert ys.kin.p.ct == pytest.approx(SAVE_S * energy / pmag)
    assert ys.ctau == pytest.approx(SAVE_S * ys.mass / pmag)


def test_helix():
    """Uniform B along z: the path length along a helix is exact"""
    bz = 1.0 * u.tesla
    pt, pz = 100.0 * u.MeV, 50.0 * u.MeV
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make(z=bz))
    ys = _solve(field, Cartesian3.make(x=pt, z=pz))
    pmag = jnp.hypot(pt, pz)
    radius = pt / (u.c_light * bz)
    # chord length in the transverse plane after turning by s_t / R
    s_t = SAVE_S * pt / pmag
    chord = 2 * radius * jnp.abs(jnp.sin(s_t / (2 * radius)))
    assert jnp.hypot(ys.kin.p.x, ys.kin.p.y) == pytest.approx(chord, rel=1e-4)
    assert ys.kin.p.z == pytest.approx(SAVE_S * pz / pmag, rel=1e-5)
    assert abs(ys.kin.t.to_cartesian3()) == pytest.approx(
        jnp.full_like(SAVE_S, pmag), rel=1e-6
    )
    assert ys.ctau == pytest.approx(SAVE_S * ys.mass / pmag, rel=1e-6)
    # proper time and lab time are related by gamma
    assert ys.kin.p.ct == pytest.approx(ys.gamma() * ys.ctau, rel=1e-6)


def test_ctau_gradient():
    """d(ctau)/d|p| = -s m / |p|^2 in a field-free region"""
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())

    def final_ctau(pz: float) -> float:
        return _solve(field, Cartesian3.make(z=pz)).ctau[-1]

    pz = 200.0 * u.MeV
    mass = MuonStateDs.make(
        position=Cartesian4.make(), momentum=Cartesian3.make(z=pz), q=1
    ).mass
    grad = jax.jacfwd(final_ctau)(pz)
    assert grad == pytest.approx(-SAVE_S[-1] * mass / pz**2, rel=1e-6)
