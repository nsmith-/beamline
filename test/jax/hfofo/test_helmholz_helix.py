from pathlib import Path

import hepunits as u
import jax.numpy as jnp
import pytest

pytest.importorskip("pxr", reason="usd-core not installed")


from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.export.usd import add_trajectories, add_volume, make_stage, save_usdz
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDct
from beamline.jax.magnet.loop import WireLoop


def test_helmholz_helix(artifacts_dir: Path):
    """A double-helix like structure of helmholz coils

    The goal is to see if this can match the dipole field pattern
    of the helical FOFO lattice (which nominally uses tilted solenoids)
    """
    path = str(artifacts_dir / "helmholz_helix.usdz")
    stage = make_stage(path)

    loop = WireLoop(
        R=250.0 * u.mm,
        I=50.0 * u.A,
    )
    add_volume(stage, "/beamline/loop", loop)

    field = loop
    start = MuonStateDct.make(
        position=Cartesian4.make(z=-800.0 * u.mm),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    cts = jnp.linspace(0.0, 3.0 * u.m, 100)
    states, _ = diffrax_solve(field, start, cts)
    add_trajectories(stage, "/trajectories", states, animate_marker=True)

    save_usdz(stage)
