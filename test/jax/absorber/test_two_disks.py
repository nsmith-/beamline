"""
Muon beam through two touching SiO2 disks.

Two solid SiO2 disks back-to-back along z (touching, no field-free
gap). A particle: (1) gets energy loss + multiple scattering in disk 1, applied
in place; (2) drifts through disk 2's thickness on its post-disk-1 direction, so
disk-1's exit angle contributes to the final transverse position; (3) gets
energy loss + scattering in disk 2.
"""

from __future__ import annotations

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.absorber.absorber import CylindricalAbsorber
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.kinematics import MuonStateDz

# dist_stats lives at test/dist_stats.py (see test_absorber.py note).
import dist_stats as ds

# Uncomment to move this file into the slow "extended" set:
# pytestmark = pytest.mark.extended

# --- configuration -----------------------------------------------------------
MATERIAL = "silicon_dioxide_SiO2"
BEAM_PC = 200.0  # MeV/c
RADIUS = 100.0  # mm
DISK1_LEN = 5.0  # mm
DISK2_LEN = 5.0  # mm
N_PARTICLES = 1_000_000
SEED = 7
N_BOOT = 500
N_BINS = 2000
N_BINS_MCS = 200

# --- tolerances --------------------------------------------------------------
# !!! PLACEHOLDERS -- tune against a real run (`-v -s`).
MODE_RTOL = 0.03  # one-disk fitted mode vs predicted MPV
ANGLE_RTOL = 0.03  # two-disk theta_x RMS vs its two-independent-kick model
DX_RTOL = 0.05  # two-disk dx RMS vs the 8/3 touching-disks theory
# Two-disk angle relative to the single-10mm-slab truth: the two-independent-
# kick sim is ~3% low (Highland log term), so expect the ratio in this band.
THETA0_10MM_BAND = (0.95, 1.00)


def make_muon(pc_MeV):
    return MuonStateDz.make(
        position=Cartesian4.make(z=0.0),
        momentum=Cartesian3.make(z=pc_MeV * u.MeV),
        q=1,
    )


def drift_through_thickness(state, thickness):
    """Advance transverse position by theta * thickness (small-angle drift).

    This is the inter-disk longitudinal propagation that CylindricalAbsorber.apply
    does not perform; it carries disk-1's exit angle across disk-2's thickness.
    """
    px, py, pz = state.kin.t.x, state.kin.t.y, state.kin.t.z
    new_position = Cartesian4.make(
        x=state.kin.p.x + px * thickness / pz,
        y=state.kin.p.y + py * thickness / pz,
        z=state.kin.p.z,
        ct=state.kin.p.ct,
    )
    new_kin = Tangent(p=new_position, t=state.kin.t)
    return eqx.tree_at(lambda s: s.kin, state, new_kin)


@pytest.fixture(scope="module")
def simulation():
    """Run both the two-disk and single-disk beams once; share results."""
    mat = MATERIALS[MATERIAL]
    disk1 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK1_LEN * u.mm)
    disk2 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK2_LEN * u.mm)

    def through_two_disks(state, key):
        state, key = disk1.apply(state, key)
        state = drift_through_thickness(state, disk2.length)
        state, key = disk2.apply(state, key)
        return state

    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    out = jax.jit(jax.vmap(through_two_disks))(beam, keys)

    dE_total = np.asarray(beam.kin.t.ct - out.kin.t.ct)
    pc_out = np.asarray(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))
    theta_x_two = np.arctan2(np.asarray(out.kin.t.x), np.asarray(out.kin.t.z))
    dx_two = np.asarray(out.kin.p.x - beam.kin.p.x)

    # single disk alone (for the overlays + a one-disk consistency check)
    out1, _ = jax.jit(jax.vmap(disk1.apply))(beam, keys)
    dE_one = np.asarray(beam.kin.t.ct - out1.kin.t.ct)
    theta_x_one = np.arctan2(np.asarray(out1.kin.t.x), np.asarray(out1.kin.t.z))
    dx_one = np.asarray(out1.kin.p.x - beam.kin.p.x)

    # predictions
    probe = make_muon(BEAM_PC)
    pp1 = mat.straggling_params(probe, disk1.length)
    theta0_one = float(mat.highland_theta0(probe, disk1.length))
    theta0_10mm = float(mat.highland_theta0(probe, disk1.length + disk2.length))
    y_rms_one_pred = float(disk1.length) * theta0_one / np.sqrt(3.0)
    theta0_two_disk = np.sqrt(2.0) * theta0_one  # two-independent-kick (quadrature)
    x1, x2 = float(disk1.length), float(disk2.length)
    y_rms_two_touching = float(np.sqrt((8.0 / 3.0) * x1 * x2 * theta0_one**2))
    y_rms_10mm = (x1 + x2) * theta0_10mm / np.sqrt(3.0)  # single-slab (PDG-correct)

    hr1 = (0.0, float(np.percentile(dE_one, 99.5)))
    hr2 = (0.0, float(np.percentile(dE_total, 99.5)))
    stats1 = ds.summarize(dE_one, name="one 5 mm disk", n_boot=N_BOOT,
                          bins=N_BINS, hist_range=hr1, seed=SEED)
    stats2 = ds.summarize(dE_total, name="two disks", n_boot=N_BOOT,
                          bins=N_BINS, hist_range=hr2, seed=SEED)

    return {
        "pp1": pp1,
        "theta0_one": theta0_one,
        "theta0_two_disk": theta0_two_disk,
        "theta0_10mm": theta0_10mm,
        "y_rms_one_pred": y_rms_one_pred,
        "y_rms_two_touching": y_rms_two_touching,
        "y_rms_10mm": y_rms_10mm,
        "dE_one": dE_one,
        "dE_total": dE_total,
        "pc_out": pc_out,
        "theta_x_one": theta_x_one,
        "theta_x_two": theta_x_two,
        "dx_one": dx_one,
        "dx_two": dx_two,
        "stats1": stats1,
        "stats2": stats2,
        "hr1": hr1,
        "hr2": hr2,
        "material_name": mat.name,
    }


def test_one_disk_mode(simulation):
    """Single-disk fitted Landau mode matches the predicted MPV."""
    assert simulation["stats1"]["mode"] == pytest.approx(
        float(simulation["pp1"].mode_energy_loss), rel=MODE_RTOL
    )


def test_two_disks_lose_more(simulation):
    """Two disks lose more energy (larger fitted mode) than one."""
    assert simulation["stats2"]["mode"] > simulation["stats1"]["mode"]


def test_momentum_is_degraded(simulation):
    assert simulation["pc_out"].mean() < BEAM_PC


def test_one_disk_scattering(simulation):
    """The single-disk building block reproduces Highland theta0 and dx RMS."""
    assert float(np.std(simulation["theta_x_one"])) == pytest.approx(
        simulation["theta0_one"], rel=ANGLE_RTOL
    )
    assert float(np.std(simulation["dx_one"])) == pytest.approx(
        simulation["y_rms_one_pred"], rel=DX_RTOL
    )


def test_two_disk_scattering_angle(simulation):
    """Two-disk theta_x RMS matches the two-independent-kick model, and is
    ~3% below the single-10mm-slab truth (documented Highland-log deficit)."""
    theta_x_rms = float(np.std(simulation["theta_x_two"]))
    # (a) the sim reproduces its own construction: sqrt(2)*theta0(5mm)
    assert theta_x_rms == pytest.approx(simulation["theta0_two_disk"], rel=ANGLE_RTOL)
    # (b) that value sits just below the PDG-correct single-slab theta0(10mm)
    ratio = theta_x_rms / simulation["theta0_10mm"]
    lo, hi = THETA0_10MM_BAND
    assert lo <= ratio <= hi, (
        f"two-disk theta_x RMS / theta0(10mm) = {ratio:.4f} outside {THETA0_10MM_BAND}; "
        "the two-independent-kick model should be a few % below the single-slab value"
    )


def test_two_disk_displacement(simulation):
    """Two-disk dx RMS matches the 8/3 touching-disks theory (drift + cross-term),
    which is ~2x the naive quadrature of the two in-disk offsets."""
    dx_rms = float(np.std(simulation["dx_two"]))
    assert dx_rms == pytest.approx(simulation["y_rms_two_touching"], rel=DX_RTOL)
    # sanity: far above quadrature of two in-disk offsets (= sqrt(2/3) x theta0)
    x = float(DISK1_LEN)
    quadrature_only = np.sqrt(2.0 / 3.0) * x * simulation["theta0_one"]
    assert dx_rms > 1.5 * quadrature_only


def test_summary_figure(simulation, artifacts_dir):
    """Render the four-panel validation figure into test_artifacts/."""
    s = simulation
    dE_one, dE_total, pc_out = s["dE_one"], s["dE_total"], s["pc_out"]
    theta_x_one, theta_x_two = s["theta_x_one"], s["theta_x_two"]
    dx_one, dx_two = s["dx_one"], s["dx_two"]
    theta0_two_disk = s["theta0_two_disk"]
    theta0_10mm = s["theta0_10mm"]
    y_rms_two_touching = s["y_rms_two_touching"]
    y_rms_10mm = s["y_rms_10mm"]
    stats2, hr2 = s["stats2"], s["hr2"]
    fit2 = stats2["_fit"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axL, axR), (axT, axY) = axes
    fig.suptitle(
        f"{BEAM_PC:.0f} MeV/c muons through two {DISK1_LEN:.0f} mm "
        f"{s['material_name']} disks (touching)",
        fontsize=13,
        fontweight="bold",
    )

    # top-left: energy-loss spectrum (one disk vs both)
    hiA = hr2[1]
    axL.hist(dE_one, bins=N_BINS, range=(0, hiA), density=True,
             color="#c7c7c7", alpha=0.9, label=f"one {DISK1_LEN:.0f} mm disk")
    axL.hist(dE_total, bins=N_BINS, range=(0, hiA), density=True,
             histtype="step", lw=2, color="#4c72b0", label="both disks (summed)")
    norm = len(dE_total) * hr2[1] / N_BINS
    xx = np.linspace(fit2["fit_lo"], fit2["fit_hi"], 300)
    axL.plot(xx, ds._gaussian(xx, *fit2["popt"]) / norm, color="k", lw=2,
             label=f"two-disk peak fit\nmode = {stats2['mode']:.3f} "
                   f"$\\pm$ {stats2['mode_err']:.3f} MeV")
    axL.set(xlabel="energy loss $\\Delta E$ [MeV]", ylabel="probability density",
            title="Landau energy-loss spectrum")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)
    axL.set_xlim(0, stats2["mode"] + 8 * fit2["sigma"])

    # top-right: outgoing momentum
    axR.hist(pc_out, bins=N_BINS, range=(np.percentile(pc_out, 0.5), BEAM_PC),
             color="#8172b3", alpha=0.85, density=True)
    axR.axvline(BEAM_PC, color="0.3", lw=2, ls=":", label=f"incoming {BEAM_PC:.0f} MeV/c")
    axR.set(xlabel="outgoing momentum |p| [MeV/c]", ylabel="probability density",
            title="Momentum after both disks")
    axR.legend()
    axR.grid(alpha=0.3)

    # bottom-left: angular deflection theta_x (two reference lines)
    t_lim = 5.0 * theta0_two_disk
    axT.hist(theta_x_one, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(theta_x_one) * 1e3:.3f} mrad)")
    axT.hist(theta_x_two, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             histtype="step", lw=2, color="#937860", density=True,
             label=f"two disks (RMS = {np.std(theta_x_two) * 1e3:.3f} mrad)")
    th = np.linspace(-t_lim, t_lim, 400)
    axT.plot(th, np.exp(-0.5 * (th / theta0_two_disk) ** 2)
             / (theta0_two_disk * np.sqrt(2 * np.pi)), color="k", lw=2,
             label=f"two-independent-kick ($\\sqrt{{2}}\\,\\theta_0$ = "
                   f"{theta0_two_disk * 1e3:.3f} mrad)")
    axT.plot(th, np.exp(-0.5 * (th / theta0_10mm) ** 2)
             / (theta0_10mm * np.sqrt(2 * np.pi)), color="#c44e52", lw=1.5, ls="--",
             label=f"single-10mm Highland ($\\theta_0$ = {theta0_10mm * 1e3:.3f} mrad, "
                   f"PDG-correct)")
    axT.set(xlabel="$\\theta_x$ [rad]", ylabel="probability density",
            title="Angular deflection (one plane)")
    axT.legend(fontsize=8)
    axT.grid(alpha=0.3)

    # bottom-right: lateral displacement dx (two reference lines)
    y_lim = 5.0 * y_rms_two_touching
    axY.hist(dx_one, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(dx_one) * 1e3:.3f} mm $\\times 10^{{-3}}$)")
    axY.hist(dx_two, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             histtype="step", lw=2, color="#da8bc3", density=True,
             label=f"two disks (RMS = {np.std(dx_two) * 1e3:.3f} mm $\\times 10^{{-3}}$)")
    yy = np.linspace(-y_lim, y_lim, 400)
    axY.plot(yy, np.exp(-0.5 * (yy / y_rms_two_touching) ** 2)
             / (y_rms_two_touching * np.sqrt(2 * np.pi)), color="k", lw=2,
             label=f"two-disk touching theory ({y_rms_two_touching * 1e3:.3f} $\\mu$m)")
    axY.plot(yy, np.exp(-0.5 * (yy / y_rms_10mm) ** 2)
             / (y_rms_10mm * np.sqrt(2 * np.pi)), color="#c44e52", lw=1.5, ls="--",
             label=f"single-10mm slab ({y_rms_10mm * 1e3:.3f} $\\mu$m, PDG-correct)")
    axY.set(xlabel="lateral displacement $\\Delta x$ [mm]", ylabel="probability density",
            title="Lateral displacement")
    axY.legend(fontsize=8)
    axY.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(artifacts_dir / "two_disks_simulation.png", dpi=130)
    plt.close(fig)
