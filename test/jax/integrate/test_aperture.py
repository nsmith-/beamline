"""Tests for aperture loss handling in the solvers"""

import hepunits as u
import jax.numpy as jnp
import jax.random as jr
import pytest

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.geometry import BeamPipe
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.integrate.stochastic import stochastic_solve
from beamline.jax.kinematics import MuonStateDz

PIPE = BeamPipe(radius=50.0 * u.mm)
SAVE_Z = jnp.linspace(0.0, 1000.0 * u.mm, 11)
# in a field-free region, x = x0 + (px / pz) z
X0, PX, PZ = 12.0 * u.mm, 10.0 * u.MeV, 200.0 * u.MeV
EXIT_Z = (PIPE.radius - X0) * PZ / PX  # 760 mm


def _free_field() -> SimpleEMField:
    return SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())


def _muon(px: float) -> MuonStateDz:
    return MuonStateDz.make(
        position=Cartesian4.make(x=X0),
        momentum=Cartesian3.make(x=px, z=PZ),
        q=1,
    )


def _check_lost(ys: MuonStateDz) -> None:
    alive = ys.is_alive()
    assert jnp.all(alive == (SAVE_Z < EXIT_Z))
    # frozen at the exit point, just past the wall
    lost_at = ys.lost_at[-1]
    assert lost_at == pytest.approx(EXIT_Z, abs=1.0 * u.mm)
    assert jnp.all(ys.lost_at[~alive] == lost_at)
    assert jnp.allclose(ys.kin.p.z[~alive], lost_at)
    assert jnp.all(ys.kin.p.x[~alive] >= PIPE.radius)
    # before loss, the straight-line track is unaffected
    expected_x = X0 + PX / PZ * SAVE_Z
    assert ys.kin.p.x[alive] == pytest.approx(expected_x[alive])


def test_diffrax_solve_aperture():
    ys, stats = diffrax_solve(_free_field(), _muon(PX), SAVE_Z, aperture=PIPE)
    assert stats["lost"]
    _check_lost(ys)


def test_diffrax_solve_aperture_survives():
    ys, stats = diffrax_solve(_free_field(), _muon(0.0), SAVE_Z, aperture=PIPE)
    assert not stats["lost"]
    assert jnp.all(ys.is_alive())
    assert ys.kin.p.z == pytest.approx(SAVE_Z)


def test_stochastic_solve_aperture():
    # small absorber off to the side of the track, so only the aperture matters
    absorber = AbsorberCylinder(
        material=MATERIALS["lithium_hydride_LiH"],
        radius=5.0 * u.mm,
        length=10.0 * u.mm,
        char_length=10.0 * u.mm,
    )
    ys, _ = stochastic_solve(
        _free_field(), absorber, _muon(PX), SAVE_Z, jr.key(0), aperture=PIPE
    )
    alive = ys.is_alive()
    assert jnp.all(alive == (SAVE_Z < EXIT_Z))
    # lost after the first accepted step that ends outside, and frozen there
    lost_at = ys.lost_at[-1]
    assert EXIT_Z <= lost_at < EXIT_Z + 50.0 * u.mm
    assert jnp.all(ys.lost_at[~alive] == lost_at)
    assert jnp.allclose(ys.kin.p.z[~alive], lost_at)
    assert jnp.all(ys.kin.p.x[~alive] >= PIPE.radius)
