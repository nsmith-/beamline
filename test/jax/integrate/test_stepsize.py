"""Tests for the boundary-aware step size controller"""

import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.geometry import BeamPipe
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDz


def test_start_inside_volume():
    """Starting inside a volume (negative signed distance) must still step forward

    Regression test: the controller's initial step was clipped by the *signed*
    distance, so a particle starting deep inside a volume (here the aperture, which
    contributes to the signed distance) took its first step backwards.
    """
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    px, pz = 10.0 * u.MeV, 200.0 * u.MeV
    start = MuonStateDz.make(
        position=Cartesian4.make(x=1.0 * u.mm),
        momentum=Cartesian3.make(x=px, z=pz),
        q=1,
    )
    save_z = jnp.linspace(0.0, 500.0 * u.mm, 6)
    # the track stays well inside the pipe
    ys, stats = diffrax_solve(
        field, start, save_z, aperture=BeamPipe(radius=100.0 * u.mm)
    )
    assert not stats["lost"]
    assert ys.kin.p.z == pytest.approx(save_z)
    assert ys.kin.p.x == pytest.approx(1.0 * u.mm + px / pz * save_z)
