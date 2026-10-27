"""
Muon beam through a single SiO2 absorber.

  * energy loss     -- fitted Landau mode vs the predicted most-probable value
  * energy loss     -- dE matches scipy.stats.landau at the mapped (loc, scale)
  * momentum        -- outgoing |p| is degraded relative to the incoming beam
  * scattering      -- bulk theta_x width vs Highland theta_0 (PDG 34.16, 34.3)
  * scattering      -- raw std inflated by the unbounded Landau tail

The beam is propagated with ``stochastic_solve``; all physics lives in the
library. The integrator segments the crossing into several PID-chosen
sub-steps, so the multiple-scattering width comes in a few percent under
single-application Highland (the quadrature deficit, PDG 34.3).
"""

import dist_stats as ds
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy import stats as sps

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.scattering import highland_scattering_sampler
from beamline.jax.absorber.straggling import (
    _straggling_to_landau,
    landau_energy_loss_sampler,
)
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrate.stochastic import (
    StochasticKick,
    energy_loss_kick,
    scattering_kick,
    stochastic_solve,
)
from beamline.jax.kinematics import MuonStateDz, TrackingStatus
from beamline.units import MUON_MASS

# --- configuration -----------------------------------------------------------
MATERIAL = "silicon_dioxide_SiO2"
BEAM_PC = 200.0 * u.MeV
RADIUS = 100.0 * u.mm
LENGTH = 10.0 * u.mm
START_Z = -20.0 * u.mm
END_Z = 20.0 * u.mm
N_PARTICLES = 10_000
SEED = 42
N_BOOT = 500  # bootstrap resamples for the uncertainties
N_BINS = 2000  # histogram bins for the spectrum + Gaussian peak fit
N_BINS_MCS = 200  # histogram bins for the scattering observables

# --- tolerances (tuned against real runs) ------------------------------------
MODE_RTOL = 0.02  # fitted Landau mode vs predicted MPV
THETA0_RTOL = 0.02  # empirical theta RMS vs Highland theta0
# theta_x width / theta0(10 mm): low because the crossing is segmented into
# sub-steps whose Highland kicks add in quadrature (see test_scattering_angle)
THETA_RATIO_RANGE = (0.88, 1.02)


def make_absorber(char_length: float = LENGTH) -> AbsorberCylinder:
    """A SiO2 disk centred at the origin, axis along z; char_length caps the in-material step."""
    return AbsorberCylinder(
        material=MATERIALS[MATERIAL],
        radius=RADIUS,
        length=LENGTH,
        char_length=char_length,
    )


def make_muon() -> MuonStateDz:
    """A +1 muon on-axis upstream of the absorber, travelling along +z."""
    return MuonStateDz.make(
        position=Cartesian4.make(z=START_Z),
        momentum=Cartesian3.make(z=BEAM_PC),
        q=1,
    )


def run_beam(absorber, start):
    """Propagate an ensemble through ``absorber``; return the saved states.

    Single-interval save grid [START_Z, END_Z]: an interior save point would
    force a sub-interval boundary inside the absorber and segment the traversal.
    """
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    zs = jnp.array([START_Z, END_Z])
    kick = StochasticKick(
        straggling=energy_loss_kick(landau_energy_loss_sampler),
        scattering=scattering_kick(highland_scattering_sampler),
    )
    run = jax.jit(
        jax.vmap(
            lambda k: stochastic_solve(field, absorber, start, zs, k, kick=kick)[0]
        )
    )
    return run(jr.split(jr.key(SEED), N_PARTICLES))


@pytest.fixture(scope="module")
def simulation():
    """Run the beam once and expose observables + predictions to all tests."""
    absorber = make_absorber()
    start = make_muon()
    params = absorber.interaction_params(start, LENGTH)

    ys = run_beam(absorber, start)
    energy_in = float(start.kin.t.ct)
    dE = np.asarray(energy_in - ys.kin.t.ct[:, -1])
    pc_out = np.asarray(jnp.sqrt(jnp.sum(ys.kin.t.coords[:, -1, :3] ** 2, axis=-1)))
    theta_x = np.arctan2(np.asarray(ys.kin.t.x[:, -1]), np.asarray(ys.kin.t.z[:, -1]))

    hr = (0.0, float(np.percentile(dE, 99.5)))
    stats = ds.summarize(
        dE,
        name=f"dE ({LENGTH / u.mm:.0f} mm {absorber.material.name})",
        n_boot=N_BOOT,
        bins=N_BINS,
        hist_range=hr,
        seed=SEED,
    )

    return {
        "pp": params,
        "theta0": float(params.theta0),
        "pc_in": float(jnp.sqrt(jnp.sum(start.kin.t.coords[:3] ** 2))),
        "pc_out": pc_out,
        "dE": dE,
        "theta_x": theta_x,
        "status": np.asarray(ys.status[:, -1]),
        "stats": stats,
        "hist_range": hr,
        "material_name": absorber.material.name,
    }


def test_energy_loss_mode(simulation):
    """The fitted Landau peak matches the predicted most-probable energy loss."""
    fitted_mode = simulation["stats"]["mode"]
    predicted_mode = float(simulation["pp"].mode_energy_loss)
    assert fitted_mode == pytest.approx(predicted_mode, rel=MODE_RTOL)


def test_energy_loss_distribution(simulation):
    """dE matches scipy.stats.landau at the mapped (loc, scale).

    Mirrors test_straggling.py's KS test.
    """
    loc, scale = (float(v) for v in _straggling_to_landau(simulation["pp"]))
    # Subsample: a KS test on the full ensemble rejects on negligible
    # deviations (e.g. the rest-mass floor in apply_energy_loss).
    rng = np.random.default_rng(SEED)
    n = min(2_000, len(simulation["dE"]))
    sample = rng.choice(simulation["dE"], size=n, replace=False)
    ks = sps.kstest(sample, lambda x: sps.landau.cdf(x, loc=loc, scale=scale))
    assert ks.pvalue > 0.05, f"dE does not match scipy.stats.landau: {ks}"


def test_momentum_is_degraded(simulation):
    """Passing through the absorber reduces the beam momentum."""
    assert simulation["pc_out"].mean() < simulation["pc_in"]


def test_scattering_angle(simulation):
    """Bulk theta_x width vs single-application Highland theta_0 (10 mm).

    The integrator segments the crossing into several PID-chosen sub-steps,
    each taking an independent Highland kick. Per PDG 34.3 the per-segment
    widths combine in quadrature systematically low relative to one application
    over the full thickness.
    """
    # Bulk width: the median absolute deviation, scaled to equal the standard
    # deviation for a Gaussian (scale="normal" multiplies by 1/Phi^-1(3/4) =
    # 1.4826). Insensitive to the single-scattering tail Highland ignores.
    width = sps.median_abs_deviation(simulation["theta_x"], scale="normal")
    ratio = float(width) / simulation["theta0"]
    print(f"  bulk theta_x width / theta0(10mm) = {ratio:.4f}")  # noqa: T201
    lo, hi = THETA_RATIO_RANGE
    assert lo < ratio < hi, f"bulk width / theta0(10mm) = {ratio:.4f}"


def test_scattering_tail(simulation):
    """Muons that range out are stopped, not scattered by a divergent angle.

    Rare Landau draws exceed the muon's kinetic energy. Before tracking status
    existed, apply_energy_loss floored the energy at the rest mass and
    Highland's 1/(beta*p) diverged, inflating the raw std of the angle well
    above the bulk width. Now such muons are marked STOPPED_IN_MATERIAL and
    frozen, so the surviving muons have no divergent tail: their raw RMS
    (not just the tail-insensitive bulk width) matches Highland's theta0 to
    the same tolerance as test_scattering_angle.
    """
    status = simulation["status"]
    stopped = status == TrackingStatus.STOPPED_IN_MATERIAL
    assert np.all(stopped | (status == TrackingStatus.ALIVE))
    print(f"  stopped fraction = {stopped.mean():.4f}")  # noqa: T201
    assert stopped.mean() < 0.01
    ratio = float(np.std(simulation["theta_x"][~stopped])) / simulation["theta0"]
    print(f"  surviving theta_x RMS / theta0(10mm) = {ratio:.4f}")  # noqa: T201
    lo, hi = THETA_RATIO_RANGE
    assert lo < ratio < hi, f"surviving RMS / theta0(10mm) = {ratio:.4f}"


def test_summary_figure(simulation, artifacts_dir):
    """Render the three-panel validation figure into test_artifacts/."""
    s = simulation
    dE, pc_out, theta_x = s["dE"], s["pc_out"], s["theta_x"]
    theta0 = s["theta0"]
    stats, pp, hr = s["stats"], s["pp"], s["hist_range"]
    fit = stats["_fit"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axL, axR, axT = axes
    fig.suptitle(
        f"{BEAM_PC / u.MeV:.0f} MeV/c muon beam through "
        f"{LENGTH / u.mm:.0f} mm {s['material_name']}",
        fontsize=13,
        fontweight="bold",
    )

    # left: energy-loss spectrum
    axL.hist(dE, bins=N_BINS, range=hr, color="#4c72b0", alpha=0.85, density=True)
    norm = len(dE) * (hr[1] - hr[0]) / N_BINS
    xx = np.linspace(fit["fit_lo"], fit["fit_hi"], 300)
    axL.plot(
        xx,
        ds._gaussian(xx, *fit["popt"]) / norm,
        color="k",
        lw=2,
        label=f"Gaussian peak fit\nmode = {stats['mode']:.3f} $\\pm$ "
        f"{stats['mode_err']:.3f} MeV",
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

    # middle: outgoing momentum
    axR.hist(
        pc_out,
        bins=N_BINS,
        range=(np.percentile(pc_out, 0.5), BEAM_PC / u.MeV),
        color="#8172b3",
        alpha=0.85,
        density=True,
    )
    axR.axvline(
        BEAM_PC / u.MeV,
        color="0.3",
        lw=2,
        ls=":",
        label=f"incoming {BEAM_PC / u.MeV:.0f} MeV/c",
    )
    axR.set(
        xlabel="outgoing momentum |p| [MeV/c]",
        ylabel="probability density",
        title="Momentum after the absorber",
    )
    axR.legend()
    axR.grid(alpha=0.3)

    # right: angular deflection theta_x
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

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(artifacts_dir / "absorber_simulation.png", dpi=130)
    plt.close(fig)


@pytest.mark.parametrize(
    ("pc", "pdg_range"),
    # https://pdg.lbl.gov/2025/AtomicNuclearProperties/MUE/muE_lithium_hydride_LiH.txt
    [
        (47.04 * u.MeV, 6.960e-1 * u.g / u.cm2),
        (85.09 * u.MeV, 4.811e0 * u.g / u.cm2),
        (199.4 * u.MeV, 4.122e1 * u.g / u.cm2),
    ],
)
def test_csda_range_vs_pdg(pc, pdg_range):
    material = MATERIALS["lithium_hydride_LiH"]
    mass = MUON_MASS * u.c_light_sq
    R = material.csda_range(pc, mass, 1.0)
    assert R == pytest.approx(pdg_range / material.density, rel=0.02)


def test_stopping_momentum():
    material = MATERIALS["lithium_hydride_LiH"]
    mass = MUON_MASS * u.c_light_sq
    p_stop = material.stopping_momentum(1.0 * u.mm, mass, 1.0)
    assert material.csda_range(p_stop, mass, 1.0) == pytest.approx(1.0 * u.mm)
    # a denser material stops the same range at higher momentum
    p_stop_al = MATERIALS["aluminum_Al"].stopping_momentum(1.0 * u.mm, mass, 1.0)
    assert p_stop_al > p_stop
