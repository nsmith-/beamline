"""Tests for USD scene export.

Requires usd-core; skipped automatically when it is not installed.
"""

import zipfile
from pathlib import Path

import hepunits as u
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from beamline.jax.absorber.scattering import highland_scattering_sampler
from beamline.jax.absorber.straggling import landau_energy_loss_sampler
from beamline.jax.integrate.stochastic import (
    StochasticKick,
    energy_loss_kick,
    scattering_kick,
    stochastic_solve,
)

pytest.importorskip("pxr", reason="usd-core not installed")

from pxr import Usd, UsdGeom, UsdLux, UsdShade

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder, TransformMaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import SimpleEMField, TransformEMField
from beamline.jax.export.usd import add_trajectories, add_volume, make_stage, save_usdz
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDct
from beamline.jax.magnet.solenoid import ThickSolenoid, ThinShellSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity


@pytest.fixture
def stage(tmp_path: Path):
    return make_stage(str(tmp_path / "test.usdz"))


def test_make_stage(stage):
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert abs(UsdGeom.GetStageMetersPerUnit(stage) - 1e-3) < 1e-10


def _mesh_radius_height(prim) -> tuple[float, float]:
    """Radius (max horizontal extent) and height (z-range) of a triangulated
    cylinder mesh, for comparing against the ABC's radius/length."""
    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    radius = np.max(np.linalg.norm(pts[:, :2], axis=-1))
    height = pts[:, 2].max() - pts[:, 2].min()
    return float(radius), float(height)


def test_add_thin_shell_solenoid(stage):
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    add_volume(stage, "/beamline/sol", sol)

    prim = stage.GetPrimAtPath("/beamline/sol")
    assert prim.IsValid()
    radius, height = _mesh_radius_height(prim)
    assert radius == pytest.approx(100.0 * u.mm, rel=1e-5)
    assert height == pytest.approx(500.0 * u.mm, rel=1e-5)


def _working_texture_dir(stage) -> Path:
    """Directory holding the stage's generated textures, in its (private,
    temporary) working directory — see :func:`make_stage`."""
    return Path(stage.GetRootLayer().realPath).parent / "textures"


def test_add_cylinder_uvs_and_texture_material(stage):
    """Cylinder meshes get a faceVarying `st` UV primvar (one value per
    face-vertex-index entry) and are bound to a UsdPreviewSurface material
    textured with a generated color-canvas PNG written next to the stage."""
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    add_volume(stage, "/beamline/sol", sol)

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/beamline/sol"))
    st_primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert st_primvar.IsDefined()
    assert st_primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying
    uvs = st_primvar.Get()
    n_face_verts = len(mesh.GetFaceVertexIndicesAttr().Get())
    assert len(uvs) == n_face_verts
    # UVs are all within the unit square.
    arr = np.array(uvs)
    assert np.all(arr >= -1e-6)
    assert np.all(arr <= 1.0 + 1e-6)

    binding_api = UsdShade.MaterialBindingAPI(mesh)
    material, _ = binding_api.ComputeBoundMaterial()
    assert material
    assert material.GetPath() == "/Materials/thin_shell_solenoid"

    texture_path = _working_texture_dir(stage) / "thin_shell_solenoid.png"
    assert texture_path.exists()


def test_add_sum_field_shares_texture_across_instances(stage):
    """Two instances of the same concrete type share one generated texture
    file and one Material prim rather than duplicating both per instance."""
    cav1 = PillboxCavity(
        length=200.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    cav2 = PillboxCavity(
        length=250.0 * u.mm,
        frequency=1.3 * u.GHz,
        E0=20.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    add_volume(stage, "/beamline/combined", cav1 + cav2)

    mesh0 = UsdGeom.Mesh(stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_0"))
    mesh1 = UsdGeom.Mesh(stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_1"))
    mat0, _ = UsdShade.MaterialBindingAPI(mesh0).ComputeBoundMaterial()
    mat1, _ = UsdShade.MaterialBindingAPI(mesh1).ComputeBoundMaterial()
    assert mat0.GetPath() == mat1.GetPath() == "/Materials/pillbox_cavity"

    texture_dir = _working_texture_dir(stage)
    assert [p.name for p in texture_dir.glob("pillbox_cavity*.png")] == [
        "pillbox_cavity.png"
    ]


def test_make_stage_requires_usdz_extension(tmp_path: Path):
    with pytest.raises(ValueError, match="usdz"):
        make_stage(str(tmp_path / "scene.usda"))


def test_save_usdz_bundles_textures_and_removes_workdir(stage):
    """save_usdz packages the stage and its generated textures into a single
    self-contained .usdz at the path given to make_stage, and cleans up the
    temporary working directory it was authored in."""
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    add_volume(stage, "/beamline/sol", sol)

    usdz_path = stage._beamline_usdz_path
    workdir = stage._beamline_workdir
    save_usdz(stage)

    assert Path(usdz_path).exists()
    assert not Path(workdir).exists()

    with zipfile.ZipFile(usdz_path) as z:
        names = z.namelist()
    assert any(n.endswith(".usda") for n in names)
    assert any(n == "textures/thin_shell_solenoid.png" for n in names)

    loaded = Usd.Stage.Open(usdz_path)
    assert loaded.GetPrimAtPath("/beamline/sol").IsValid()


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
    radius, height = _mesh_radius_height(prim)
    assert height == pytest.approx(float(cav.length), rel=1e-5)
    assert radius == pytest.approx(float(cav.radius), rel=1e-5)


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

    assert stage.GetPrimAtPath("/beamline/combined/thin_shell_solenoid_0").IsValid()
    assert stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_1").IsValid()


def test_add_sum_field_distinguishes_same_type_components(stage):
    """Two cavities of the same concrete type still get distinct prim names."""
    cav1 = PillboxCavity(
        length=200.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    cav2 = PillboxCavity(
        length=250.0 * u.mm,
        frequency=1.3 * u.GHz,
        E0=20.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.0,
    )
    combined = cav1 + cav2
    add_volume(stage, "/beamline/combined", combined)

    prim0 = stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_0")
    prim1 = stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_1")
    assert prim0.IsValid()
    assert prim1.IsValid()
    _, height0 = _mesh_radius_height(prim0)
    _, height1 = _mesh_radius_height(prim1)
    assert height0 != height1


def test_add_sum_field_transformed_same_type_components(stage):
    """Two identically-typed cavities placed via a Transform inside a SumField
    still get distinct prim names, based on the innermost type rather than
    the wrapper, and each is baked into its own global position — reproducing
    (and verifying the fix for) the scenario where viewers that name objects
    from the leaf prim only (e.g. Blender) collided two same-named leaves
    that used to differ only by their enclosing Xform's name."""
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
    placed1 = TransformEMField(
        transform=Transform.make_translation(z=-500.0 * u.mm), field=cav
    )
    placed2 = TransformEMField(
        transform=Transform.make_translation(z=500.0 * u.mm), field=cav
    )
    combined = placed1 + placed2
    add_volume(stage, "/beamline/combined", combined)

    prim0 = stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_0")
    prim1 = stage.GetPrimAtPath("/beamline/combined/pillbox_cavity_1")
    assert prim0.IsValid()
    assert prim1.IsValid()
    assert prim0.IsA(UsdGeom.Mesh)
    assert prim1.IsA(UsdGeom.Mesh)

    pts0 = np.array(UsdGeom.Mesh(prim0).GetPointsAttr().Get())
    pts1 = np.array(UsdGeom.Mesh(prim1).GetPointsAttr().Get())
    assert pts0[:, 2].mean() == pytest.approx(-500.0 * u.mm, rel=1e-5)
    assert pts1[:, 2].mean() == pytest.approx(500.0 * u.mm, rel=1e-5)


def test_concrete_types_have_unique_colors(stage):
    """Each concrete exportable type gets its own display color."""
    sol = ThinShellSolenoid(R=80.0 * u.mm, jphi=1.0, L=400.0 * u.mm)
    thick_sol = ThickSolenoid(
        Rin=60.0 * u.mm, Rout=90.0 * u.mm, jphi=1.0, L=400.0 * u.mm
    )
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
    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=350.0 * u.mm)

    add_volume(stage, "/beamline/sol", sol)
    add_volume(stage, "/beamline/thick_sol", thick_sol)
    add_volume(stage, "/beamline/cav", cav)
    add_volume(stage, "/beamline/absorber", absorber)

    colors = []
    for path in (
        "/beamline/sol",
        "/beamline/thick_sol",
        "/beamline/cav",
        "/beamline/absorber",
    ):
        prim = stage.GetPrimAtPath(path)
        gprim = UsdGeom.Gprim(prim)
        colors.append(tuple(gprim.GetDisplayColorAttr().Get()[0]))

    assert len(set(colors)) == len(colors)


def test_add_transformed_em_field(stage):
    """The transform is baked into the mesh's vertex positions (global
    coordinates), not authored as a separate Xform prim — so there's a single
    uniquely-named prim per component rather than a generically-named child
    nested under a wrapper, which is what caused viewers like Blender (which
    name objects from the leaf prim only) to collide same-named leaves."""
    sol = ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=500.0 * u.mm)
    tf = Transform.make_translation(z=1000.0 * u.mm)
    placed = TransformEMField(transform=tf, field=sol)
    add_volume(stage, "/beamline/placed", placed)

    prim = stage.GetPrimAtPath("/beamline/placed")
    assert prim.IsValid()
    assert prim.IsA(UsdGeom.Mesh)
    assert not stage.GetPrimAtPath("/beamline/placed/thin_shell_solenoid").IsValid()

    radius, height = _mesh_radius_height(prim)
    assert radius == pytest.approx(100.0 * u.mm, rel=1e-5)
    assert height == pytest.approx(500.0 * u.mm, rel=1e-5)

    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    assert pts[:, 2].mean() == pytest.approx(1000.0 * u.mm, rel=1e-5)


def test_add_transformed_material_volume(stage):
    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=350.0 * u.mm)
    tf = Transform.make_translation(z=1000.0 * u.mm)
    placed = TransformMaterialVolume(transform=tf, material=absorber)
    add_volume(stage, "/beamline/placed", placed)

    prim = stage.GetPrimAtPath("/beamline/placed")
    assert prim.IsValid()
    assert prim.IsA(UsdGeom.Mesh)
    assert not stage.GetPrimAtPath("/beamline/placed/absorber_cylinder").IsValid()

    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    assert pts[:, 2].mean() == pytest.approx(1000.0 * u.mm, rel=1e-5)


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

    # extent must be authored (required for Boundable prims; many non-usdview
    # viewers cull geometry missing it) and the trail should be dimmed.
    extent = curves.GetExtentAttr().Get()
    assert len(extent) == 2
    assert curves.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.35)

    # Curated copy for manual inspection (see AGENTS.md); built as its own
    # stage rather than saving the shared `stage` fixture, since `stage`'s
    # target path lives under `tmp_path`, not `artifacts_dir`.
    saved_stage = make_stage(str(artifacts_dir / "single_particle.usdz"))
    add_trajectories(saved_stage, "/trajectories", states)
    save_usdz(saved_stage)


def test_add_trajectories_animated_marker(stage):
    """animate_marker=True adds a moving glow-sphere marker group (not a
    UsdLux light, which Quick Look/AR Quick Look ignores) — an Xform whose
    position is time-sampled, wrapping concentric emissive Sphere shells —
    and sets the stage's time-code range to match the number of samples."""
    n_steps = 10
    positions = np.random.randn(n_steps, 4).astype(np.float32)
    positions[..., 3] = 1.0

    class _FakeKin:
        class p:
            coords = positions

    class _FakeState:
        kin = _FakeKin()

    add_trajectories(stage, "/trajectories", _FakeState(), animate_marker=True)

    marker_prim = stage.GetPrimAtPath("/trajectories/particle_0_marker")
    assert marker_prim.IsValid()
    assert marker_prim.IsA(UsdGeom.Xform)
    assert not marker_prim.IsA(UsdLux.SphereLight)

    core_prim = stage.GetPrimAtPath("/trajectories/particle_0_marker/core")
    assert core_prim.IsValid()
    assert core_prim.IsA(UsdGeom.Sphere)
    halo_prim = stage.GetPrimAtPath("/trajectories/particle_0_marker/halo1")
    assert halo_prim.IsValid()
    assert halo_prim.IsA(UsdGeom.Sphere)
    # The outer halo shell is bigger and more transparent than the core.
    core_radius = UsdGeom.Sphere(core_prim).GetRadiusAttr().Get()
    halo_radius = UsdGeom.Sphere(halo_prim).GetRadiusAttr().Get()
    assert halo_radius > core_radius
    core_opacity = UsdGeom.Gprim(core_prim).GetDisplayOpacityAttr().Get()[0]
    halo_opacity = UsdGeom.Gprim(halo_prim).GetDisplayOpacityAttr().Get()[0]
    assert core_opacity > halo_opacity

    translate_op = UsdGeom.Xformable(marker_prim).GetOrderedXformOps()[0]
    time_samples = translate_op.GetTimeSamples()
    assert time_samples == list(range(n_steps))

    first = translate_op.GetAttr().Get(Usd.TimeCode(0))
    last = translate_op.GetAttr().Get(Usd.TimeCode(n_steps - 1))
    assert tuple(first) == pytest.approx(tuple(positions[0, :3]), abs=1e-5)
    assert tuple(last) == pytest.approx(tuple(positions[-1, :3]), abs=1e-5)

    assert stage.GetStartTimeCode() == 0
    assert stage.GetEndTimeCode() == n_steps - 1


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
    """Integration test: build a simple cooling cell and export to a .usdz file."""
    path = str(artifacts_dir / "cooling_cell.usdz")
    stage = make_stage(path)

    sol = ThickSolenoid(
        Rin=250.0 * u.mm,
        Rout=419.3 * u.mm,
        jphi=500.0 * u.A / u.mm**2,
        L=140.0 * u.mm,
    )
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
    cavity_upstream = TransformEMField(transform=tf_cav, field=cav)
    add_volume(
        stage,
        "/beamline/cavity_upstream",
        cavity_upstream,
    )
    tf_cav2 = Transform.make_translation(z=650.0 * u.mm)
    cavity_downstream = TransformEMField(transform=tf_cav2, field=cav)
    add_volume(
        stage,
        "/beamline/cavity_downstream",
        cavity_downstream,
    )

    mat = MATERIALS["lithium_hydride_LiH"]
    absorber = AbsorberCylinder(material=mat, radius=150.0 * u.mm, length=150.0 * u.mm)
    add_volume(stage, "/beamline/absorber", absorber)

    field = sol + cavity_upstream + cavity_downstream
    start = MuonStateDct.make(
        position=Cartesian4.make(y=20.0 * u.mm, z=-800.0 * u.mm),
        momentum=Cartesian3.make(x=20.0 * u.MeV, z=200.0 * u.MeV),
        q=1,
    )
    cts = jnp.linspace(0.0, 3.0 * u.m, 100)
    kick = StochasticKick(
        straggling=energy_loss_kick(landau_energy_loss_sampler),
        scattering=scattering_kick(highland_scattering_sampler),
    )
    states, _ = stochastic_solve(field, absorber, start, cts, jr.key(123), kick=kick)
    add_trajectories(stage, "/trajectories", states, animate_marker=True)

    save_usdz(stage)
    assert Path(path).exists()
    # verify it round-trips
    loaded = Usd.Stage.Open(path)
    assert loaded.GetPrimAtPath("/beamline/solenoid").IsValid()
    assert loaded.GetPrimAtPath("/trajectories/particle_0").IsValid()
    assert loaded.GetPrimAtPath("/trajectories/particle_0_marker").IsValid()
    assert loaded.GetPrimAtPath("/trajectories/particle_0_marker/core").IsValid()
    assert loaded.GetEndTimeCode() == 99
