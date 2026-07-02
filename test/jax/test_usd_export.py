"""Tests for USD scene export.

Requires usd-core; skipped automatically when it is not installed.
"""

from pathlib import Path

import hepunits as u
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pxr", reason="usd-core not installed")

from pxr import Usd, UsdGeom

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder, TransformMaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import SimpleEMField, TransformEMField
from beamline.jax.export.usd import add_trajectories, add_volume, make_stage
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDct
from beamline.jax.magnet.solenoid import ThinShellSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity


@pytest.fixture
def stage(tmp_path: Path):
    return make_stage(str(tmp_path / "test.usda"))


def test_make_stage(stage):
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert abs(UsdGeom.GetStageMetersPerUnit(stage) - 1e-3) < 1e-10


def test_add_thin_shell_solenoid(stage):
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    add_volume(stage, "/beamline/sol", sol)

    prim = stage.GetPrimAtPath("/beamline/sol")
    assert prim.IsValid()
    cyl = UsdGeom.Cylinder(prim)
    assert abs(cyl.GetRadiusAttr().Get() - 100.0 * u.mm) < 1e-6
    assert abs(cyl.GetHeightAttr().Get() - 500.0 * u.mm) < 1e-6
    assert cyl.GetAxisAttr().Get() == "Z"


def test_add_pillbox_cavity(stage):
    cav = PillboxCavity(
        length=300.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    add_volume(stage, "/beamline/cav", cav)

    prim = stage.GetPrimAtPath("/beamline/cav")
    assert prim.IsValid()
    cyl = UsdGeom.Cylinder(prim)
    assert cyl.GetHeightAttr().Get() == pytest.approx(float(cav.length), rel=1e-5)
    assert cyl.GetRadiusAttr().Get() == pytest.approx(float(cav.radius), rel=1e-5)


def test_add_absorber_cylinder(stage):
    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=350.0 * u.mm)
    add_volume(stage, "/beamline/absorber", absorber)

    prim = stage.GetPrimAtPath("/beamline/absorber")
    assert prim.IsValid()


def test_add_sum_field(stage):
    sol = ThinShellSolenoid(R=80.0 * u.mm, jphi=1.0, L=400.0 * u.mm)
    cav = PillboxCavity(
        length=200.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    combined = sol + cav
    add_volume(stage, "/beamline/combined", combined)

    assert stage.GetPrimAtPath("/beamline/combined/component_0").IsValid()
    assert stage.GetPrimAtPath("/beamline/combined/component_1").IsValid()


def test_add_transformed_em_field(stage):
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    tf = Transform.make_translation(z=1000.0 * u.mm)
    placed = TransformEMField(transform=tf, field=sol)
    add_volume(stage, "/beamline/placed", placed)

    xf_prim = stage.GetPrimAtPath("/beamline/placed")
    assert xf_prim.IsValid()
    assert UsdGeom.Xform(xf_prim)
    child = stage.GetPrimAtPath("/beamline/placed/field")
    assert child.IsValid()


def test_add_transformed_material_volume(stage):
    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=350.0 * u.mm)
    tf = Transform.make_translation(z=1000.0 * u.mm)
    placed = TransformMaterialVolume(transform=tf, material=absorber)
    add_volume(stage, "/beamline/placed", placed)

    xf_prim = stage.GetPrimAtPath("/beamline/placed")
    assert xf_prim.IsValid()
    assert UsdGeom.Xform(xf_prim)
    child = stage.GetPrimAtPath("/beamline/placed/material")
    assert child.IsValid()


def test_add_trajectories_single_particle(stage, artifacts_dir: Path):
    field = SimpleEMField(
        E0=Cartesian3.make(),
        B0=Cartesian3.make(z=1.0 * u.tesla),
    )
    start = MuonStateDct.make(
        position=Cartesian4.make(y=50.0 * u.mm),
        momentum=Cartesian3.make(x=100.0 * u.MeV, z=200.0 * u.MeV),
        q=1,
    )
    cts = jnp.linspace(0.0, 2.0 * u.m, 50)
    states, _ = diffrax_solve(field, start, cts)

    add_trajectories(stage, "/trajectories", states)

    prim = stage.GetPrimAtPath("/trajectories/particle_0")
    assert prim.IsValid()
    curves = UsdGeom.BasisCurves(prim)
    pts = curves.GetPointsAttr().Get()
    assert len(pts) == 50

    stage.Save()
    saved = artifacts_dir / "single_particle.usda"
    Usd.Stage.Open(stage.GetRootLayer().realPath).Export(str(saved))


def test_add_trajectories_batch(stage):
    """Batched trajectories produce one prim per particle."""
    n_particles = 5
    n_steps = 30

    positions = np.random.randn(n_steps, n_particles, 4).astype(np.float32)
    positions[..., 3] = 1.0  # ct column, not used

    class _FakeKin:
        class p:
            coords = positions

    class _FakeState:
        kin = _FakeKin()

    add_trajectories(stage, "/batch", _FakeState())

    for i in range(n_particles):
        assert stage.GetPrimAtPath(f"/batch/particle_{i}").IsValid()


def test_unknown_volume_warns(stage):
    class _Unknown:
        pass

    with pytest.warns(UserWarning, match="unknown type"):
        add_volume(stage, "/beamline/unknown", _Unknown())


def test_full_scene(artifacts_dir: Path):
    """Integration test: build a simple cooling cell and export to a .usda file."""
    path = str(artifacts_dir / "cooling_cell.usda")
    stage = make_stage(path)

    sol = ThinShellSolenoid(R=200.0 * u.mm, jphi=100.0, L=1000.0 * u.mm)
    add_volume(stage, "/beamline/solenoid", sol)

    cav = PillboxCavity(
        length=300.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    tf_cav = Transform.make_translation(z=-650.0 * u.mm)
    add_volume(
        stage,
        "/beamline/cavity_upstream",
        TransformEMField(transform=tf_cav, field=cav),
    )
    tf_cav2 = Transform.make_translation(z=650.0 * u.mm)
    add_volume(
        stage,
        "/beamline/cavity_downstream",
        TransformEMField(transform=tf_cav2, field=cav),
    )

    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=350.0 * u.mm)
    add_volume(stage, "/beamline/absorber", absorber)

    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make(z=3.0 * u.tesla))
    start = MuonStateDct.make(
        position=Cartesian4.make(y=20.0 * u.mm),
        momentum=Cartesian3.make(x=50.0 * u.MeV, z=200.0 * u.MeV),
        q=1,
    )
    cts = jnp.linspace(0.0, 3.0 * u.m, 80)
    states, _ = diffrax_solve(field, start, cts)
    add_trajectories(stage, "/trajectories", states)

    stage.Save()
    assert Path(path).exists()
    # verify it round-trips
    loaded = Usd.Stage.Open(path)
    assert loaded.GetPrimAtPath("/beamline/solenoid").IsValid()
    assert loaded.GetPrimAtPath("/trajectories/particle_0").IsValid()
