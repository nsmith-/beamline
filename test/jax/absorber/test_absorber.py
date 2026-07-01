"""
Muon beam through a single SiO2 absorber.

  * energy loss     -- fitted Landau mode vs the predicted most-probable value
  * momentum        -- outgoing |p| is degraded relative to the incoming beam
  * scattering      -- empirical theta_x RMS vs the Highland theta_0 (PDG 34.16)
  * displacement    -- empirical dx RMS vs x*theta_0/sqrt(3) (PDG 34.20), and
                       corr(theta_x, dx) ~ sqrt(3)/2 (PDG 34.22 correlation)
"""

from __future__ import annotations

import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.absorber.absorber import CylindricalAbsorber
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.kinematics import MuonStateDz

# dist_stats lives at test/dist_stats.py (importable because test/conftest.py
# puts the test root on sys.path). Same helper the original script used.
import dist_stats as ds

# Uncomment to move this file into the slow "extended" set (excluded from the
# default `uv run pytest`; run explicitly with `uv run pytest -m extended`):
# pytestmark = pytest.mark.extended

# --- configuration -----------------------------------------------------------
MATERIAL = "silicon_dioxide_SiO2"
BEAM_PC = 200.0  # MeV/c
RADIUS = 100.0  # mm
LENGTH = 10.0  # mm
N_PARTICLES = 1_000_000
SEED = 42
N_BOOT = 500  # bootstrap resamples for the uncertainties
N_BINS = 2000  # histogram bins for the spectrum + Gaussian fit
N_BINS_MCS = 200  # histogram bins for the scattering observables

# --- tolerances --------------------------------------------------------------
# !!! PLACEHOLDERS -- tune against a real run (`-v -s`) and your physics
# judgment before relying on the test. The MCS angle is a one-sided band
# because the kick is applied on the post-energy-loss (slightly degraded)
# momentum, so the empirical RMS sits a touch ABOVE the Highland prediction.
MODE_RTOL = 0.02  # fitted Landau mode vs predicted MPV
THETA0_BAND = (1.00, 1.05)  # empirical theta_x RMS / theta0 expected here
DX_RTOL = 0.05  # empirical dx RMS vs x*theta0/sqrt(3)
CORR_ATOL = 0.02  # corr(theta_x, dx) vs sqrt(3)/2


def make_muon(pc_MeV):
    """A single +1 muon travelling along +z with momentum pc_MeV [MeV/c]."""
    return MuonStateDz.make(
        position=Cartesian4.make(z=-LENGTH / 2 * u.mm),
        momentum=Cartesian3.make(z=pc_MeV * u.MeV),
        q=1,
    )


@pytest.fixture(scope="module")
def simulation():
    """Run the beam once and expose observables + predictions to all tests."""
    absorber = CylindricalAbsorber(
        material=MATERIALS[MATERIAL], radius=RADIUS * u.mm, length=LENGTH * u.mm
    )

    # Deterministic predictions (Bethe-Bloch + Landau, Highland MCS).
    probe = make_muon(BEAM_PC)
    pp = absorber.material.straggling_params(probe, absorber.length)
    theta0 = float(absorber.material.highland_theta0(probe, absorber.length))
    y_rms_pred = float(absorber.length) * theta0 / np.sqrt(3.0)
    pc_in = float(jnp.sqrt(jnp.sum(probe.kin.t.coords[:3] ** 2)))

    # Run the beam: one PRNG key per particle.
    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    out, _ = jax.jit(jax.vmap(absorber.apply))(beam, keys)

    dE = np.asarray(beam.kin.t.ct - out.kin.t.ct)
    pc_out = np.asarray(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))
    theta_x = np.arctan2(np.asarray(out.kin.t.x), np.asarray(out.kin.t.z))
    dx = np.asarray(out.kin.p.x - beam.kin.p.x)

    hr = (0.0, float(np.percentile(dE, 99.5)))
    stats = ds.summarize(
        dE,
        name=f"dE ({LENGTH:.0f} mm {absorber.material.name})",
        n_boot=N_BOOT,
        bins=N_BINS,
        hist_range=hr,
        seed=SEED,
    )

    return {
        "pp": pp,
        "theta0": theta0,
        "y_rms_pred": y_rms_pred,
        "pc_in": pc_in,
        "pc_out": pc_out,
        "dE": dE,
        "theta_x": theta_x,
        "dx": dx,
        "stats": stats,
        "hist_range": hr,
        "material_name": absorber.material.name,
    }


def test_energy_loss_mode(simulation):
    """The fitted Landau peak matches the predicted most-probable energy loss."""
    fitted_mode = simulation["stats"]["mode"]
    predicted_mode = float(simulation["pp"].mode_energy_loss)
    assert fitted_mode == pytest.approx(predicted_mode, rel=MODE_RTOL)


def test_momentum_is_degraded(simulation):
    """Passing through the absorber reduces the beam momentum."""
    assert simulation["pc_out"].mean() < simulation["pc_in"]


def test_scattering_angle(simulation):
    """Empirical theta_x RMS sits in the expected band around Highland theta_0."""
    ratio = float(np.std(simulation["theta_x"])) / simulation["theta0"]
    lo, hi = THETA0_BAND
    assert lo <= ratio <= hi, f"theta_x RMS / theta0 = {ratio:.4f} outside {THETA0_BAND}"


def test_lateral_displacement(simulation):
    """Empirical dx RMS matches x*theta0/sqrt(3), corr(theta_x, dx) ~ sqrt(3)/2."""
    dx_rms = float(np.std(simulation["dx"]))
    assert dx_rms == pytest.approx(simulation["y_rms_pred"], rel=DX_RTOL)

    corr = float(np.corrcoef(simulation["theta_x"], simulation["dx"])[0, 1])
    assert corr == pytest.approx(np.sqrt(3) / 2, abs=CORR_ATOL)


def test_summary_figure(simulation, artifacts_dir):
    """Render the four-panel validation figure into test_artifacts/."""
    s = simulation
    dE, pc_out, theta_x, dx = s["dE"], s["pc_out"], s["theta_x"], s["dx"]
    theta0, y_rms_pred = s["theta0"], s["y_rms_pred"]
    stats, pp, hr = s["stats"], s["pp"], s["hist_range"]
    fit = stats["_fit"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axL, axR), (axT, axY) = axes
    fig.suptitle(
        f"{BEAM_PC:.0f} MeV/c muon beam through {LENGTH:.0f} mm {s['material_name']}",
        fontsize=13,
        fontweight="bold",
    )

    # top-left: energy-loss spectrum
    axL.hist(dE, bins=N_BINS, range=hr, color="#4c72b0", alpha=0.85, density=True)
    norm = len(dE) * (hr[1] - hr[0]) / N_BINS
    xx = np.linspace(fit["fit_lo"], fit["fit_hi"], 300)
    axL.plot(
        xx,
        ds._gaussian(xx, *fit["popt"]) / norm,
        color="k",
        lw=2,
        label=f"Gaussian peak fit\nmode = {stats['mode']:.3f} $\\pm$ {stats['mode_err']:.3f} MeV",
    )
    axL.axvline(
        float(pp.mode_energy_loss),
        color="#c44e52",
        lw=2,
        label=f"predicted mode = {float(pp.mode_energy_loss):.2f} MeV",
    )
    axL.axvline(
        float(pp.mean_energy_loss),
        color="#55a868",
        lw=2,
        ls="--",
        label=f"predicted mean = {float(pp.mean_energy_loss):.2f} MeV",
    )
    axL.set(
        xlabel="energy loss $\\Delta E$ [MeV]",
        ylabel="probability density",
        title="Landau energy-loss spectrum",
    )
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)
    axL.set_xlim(0, stats["mode"] + 8 * fit["sigma"])

    # top-right: outgoing momentum
    axR.hist(
        pc_out,
        bins=N_BINS,
        range=(np.percentile(pc_out, 0.5), BEAM_PC),
        color="#8172b3",
        alpha=0.85,
        density=True,
    )
    axR.axvline(BEAM_PC, color="0.3", lw=2, ls=":", label=f"incoming {BEAM_PC:.0f} MeV/c")
    axR.set(
        xlabel="outgoing momentum |p| [MeV/c]",
        ylabel="probability density",
        title="Momentum after the absorber",
    )
    axR.legend()
    axR.grid(alpha=0.3)

    # bottom-left: angular deflection theta_x
    t_lim = 5.0 * theta0
    axT.hist(
        theta_x,
        bins=N_BINS_MCS,
        range=(-t_lim, t_lim),
        color="#937860",
        alpha=0.85,
        density=True,
        label=f"empirical (RMS = {np.std(theta_x) * 1e3:.3f} mrad)",
    )
    th = np.linspace(-t_lim, t_lim, 400)
    axT.plot(
        th,
        np.exp(-0.5 * (th / theta0) ** 2) / (theta0 * np.sqrt(2 * np.pi)),
        color="k",
        lw=2,
        label=f"Highland prediction\n$\\theta_0$ = {theta0 * 1e3:.3f} mrad",
    )
    axT.set(
        xlabel="$\\theta_x$ [rad]",
        ylabel="probability density",
        title="Angular deflection (one plane)",
    )
    axT.legend(fontsize=8)
    axT.grid(alpha=0.3)

    # bottom-right: lateral displacement dx
    y_lim = 5.0 * y_rms_pred
    corr_xy = float(np.corrcoef(theta_x, dx)[0, 1])
    axY.hist(
        dx,
        bins=N_BINS_MCS,
        range=(-y_lim, y_lim),
        color="#da8bc3",
        alpha=0.85,
        density=True,
        label=f"empirical (RMS = {np.std(dx) * 1e3:.3f} mm $\\times 10^{{-3}}$)",
    )
    yy = np.linspace(-y_lim, y_lim, 400)
    axY.plot(
        yy,
        np.exp(-0.5 * (yy / y_rms_pred) ** 2) / (y_rms_pred * np.sqrt(2 * np.pi)),
        color="k",
        lw=2,
        label=f"PDG prediction\n$x\\theta_0/\\sqrt{{3}}$ = {y_rms_pred * 1e3:.3f} $\\mu$m",
    )
    axY.set(
        xlabel="lateral displacement $\\Delta x$ [mm]",
        ylabel="probability density",
        title=f"Lateral displacement (corr$(\\theta_x, \\Delta x)$ = {corr_xy:.3f})",
    )
    axY.legend(fontsize=8)
    axY.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(artifacts_dir / "absorber_simulation.png", dpi=130)
    plt.close(fig)
