"""
Simulate a muon beam passing through the SiO2 cylinder.

Each muon arrives at the absorber face and receives both an energy-loss
(sampled from the Landau distribution) and a multiple-Coulomb-scattering
deflection (sampled from the Highland formula in scattering.py). It runs
a whole beam at once with jax.vmap and looks at:

  1. The energy-loss spectrum                       (Landau)
  2. The outgoing momentum magnitude                (consequence of dE)
  3. The transverse angular deflection theta_x      (Gaussian, RMS = theta_0)
  4. The transverse lateral displacement y_x        (Gaussian, RMS = x*theta_0/sqrt(3))

For each scattering observable, the empirical histogram is overlaid with
the analytic prediction (Gaussian with the Highland-formula RMS).

This version also extracts the MODE (via a Gaussian fit to the peak), MEDIAN
and MEAN of the energy-loss spectrum, each with a 1-sigma uncertainty, using
the helper in dist_stats.py.

Run from the repo root:
    uv run python test/jax/simulate_absorber.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import hepunits as u
import matplotlib.pyplot as plt

from beamline.jax.absorber.absorber import CylindricalAbsorber
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.coordinates import Cartesian3, Cartesian4

import dist_stats as ds

# config
MATERIAL    = "silicon_dioxide_SiO2"
BEAM_PC     = 200.0      # MeV/c
RADIUS      = 100.0      # mm
LENGTH      = 10.0       # mm
N_PARTICLES = 1000000
SEED        = 42
N_BOOT      = 500        # bootstrap resamples for the uncertainties
N_BINS      = 2000       # histogram bins for the spectrum + Gaussian fit
N_BINS_MCS  = 200        # histogram bins for the scattering observables


def make_muon(pc_MeV):
    """A single +1 muon travelling along +z with momentum pc_MeV [MeV/c]."""
    return MuonStateDz.make(
        position=Cartesian4.make(z=-LENGTH / 2 * u.mm),
        momentum=Cartesian3.make(z=pc_MeV * u.MeV),
        q=1,
    )


def main():
    absorber = CylindricalAbsorber(
        material=MATERIALS[MATERIAL], radius=RADIUS * u.mm, length=LENGTH * u.mm
    )

    # Predicted straggling parameters (deterministic, from Bethe-Bloch + Landau).
    probe = make_muon(BEAM_PC)
    pp = absorber.material.straggling_params(probe, absorber.length)
    # Predicted Highland RMS plane angle and lateral offset.
    # NOTE: highland_theta0 uses the incoming kinematics; the simulation
    # applies it on the post-energy-loss momentum (slightly degraded), so
    # the empirical RMS will be very slightly larger than this prediction.
    theta0 = float(absorber.material.highland_theta0(probe, absorber.length))
    y_rms_pred = float(absorber.length) * theta0 / np.sqrt(3.0)
    E_in = float(probe.kin.t.ct)
    pc_in = float(jnp.sqrt(jnp.sum(probe.kin.t.coords[:3] ** 2)))

    # run the beam: one PRNG key per particle
    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    run = jax.jit(jax.vmap(absorber.apply))
    out, _ = run(beam, keys)

    # ---- energy-loss observables --------------------------------------------
    dE = np.asarray(beam.kin.t.ct - out.kin.t.ct)                  # MeV
    pc_out = np.asarray(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))

    # ---- scattering observables --------------------------------------------
    # Angular deflection per plane: theta_x = atan2(px, pz), small for
    # the beamline use case so this is essentially px/pz.
    px_out = np.asarray(out.kin.t.x)
    py_out = np.asarray(out.kin.t.y)
    pz_out = np.asarray(out.kin.t.z)
    theta_x = np.arctan2(px_out, pz_out)
    theta_y = np.arctan2(py_out, pz_out)

    # Lateral displacement: position changes from (0, 0) at entry to
    # (x_out, y_out) at exit, including both the MCS kick and any drift
    # implied by the model. For the thin-scatterer model in
    # scattering.py the change is exactly the y_x, y_y kick applied to
    # the entry position.
    dx = np.asarray(out.kin.p.x - beam.kin.p.x)
    dy = np.asarray(out.kin.p.y - beam.kin.p.y)

    # empirical peak of the spectrum (raw histogram argmax, for cross-check)
    h, edges = np.histogram(dE, bins=600)
    mode_emp = 0.5 * (edges[h.argmax()] + edges[h.argmax() + 1])

    # precise mode / median / mean with uncertainties
    hi = float(np.percentile(dE, 99.5))
    hr = (0.0, hi)
    stats = ds.summarize(dE, name=f"dE ({LENGTH:.0f} mm {absorber.material.name})",
                         n_boot=N_BOOT, bins=N_BINS, hist_range=hr, seed=SEED)
    fit = stats["_fit"]

    # empirical RMS of the scattering observables (the validation numbers)
    theta_x_rms = float(np.std(theta_x))
    theta_y_rms = float(np.std(theta_y))
    dx_rms = float(np.std(dx))
    dy_rms = float(np.std(dy))
    corr_xy = float(np.corrcoef(theta_x, dx)[0, 1])  # should be ~sqrt(3)/2

    print(f"Material                  : {absorber.material.name}")
    print(f"Beam                      : {BEAM_PC:.0f} MeV/c muons, N = {N_PARTICLES:,}")
    print(f"Absorber                  : {LENGTH:.0f} mm thick, R = {RADIUS:.0f} mm")
    print(f"E_in                      : {E_in:.3f} MeV   (KE = {E_in - float(probe.mass):.3f} MeV)")
    print("-" * 56)
    print("ENERGY LOSS:")
    print(f"  predicted xi              : {float(pp.xi):.4f} MeV")
    print(f"  predicted mode (Landau)   : {float(pp.mode_energy_loss):.4f} MeV")
    print(f"  empirical  mode (argmax)  : {mode_emp:.4f} MeV")
    print(f"  fitted     mode (Gauss)   : {stats['mode']:.4f} +/- {stats['mode_err']:.4f} MeV")
    print(f"  predicted mean (Bethe)    : {float(pp.mean_energy_loss):.4f} MeV")
    print(f"  empirical  mean           : {stats['mean']:.4f} +/- {stats['mean_err']:.4f} MeV")
    print(f"  momentum  {pc_in:.1f} -> {pc_out.mean():.2f} MeV/c")
    print("-" * 56)
    print("SCATTERING:")
    print(f"  predicted theta_0         : {theta0*1e3:.4f} mrad")
    print(f"  empirical  theta_x RMS    : {theta_x_rms*1e3:.4f} mrad")
    print(f"  empirical  theta_y RMS    : {theta_y_rms*1e3:.4f} mrad")
    print(f"  predicted y RMS           : {y_rms_pred*1e3:.4f} um (= x*theta_0/sqrt(3))")
    print(f"  empirical  dx RMS         : {dx_rms*1e3:.4f} um")
    print(f"  empirical  dy RMS         : {dy_rms*1e3:.4f} um")
    print(f"  correlation theta_x, dx   : {corr_xy:.4f}  (predicted {np.sqrt(3)/2:.4f})")

    # ---- plotting -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axL, axR), (axT, axY) = axes
    fig.suptitle(
        f"{BEAM_PC:.0f} MeV/c muon beam through {LENGTH:.0f} mm {absorber.material.name}",
        fontsize=13, fontweight="bold",
    )

    # --- top-left: energy-loss spectrum ---
    axL.hist(dE, bins=N_BINS, range=hr, color="#4c72b0", alpha=0.85, density=True)
    bin_width = (hr[1] - hr[0]) / N_BINS
    norm = len(dE) * bin_width
    xx = np.linspace(fit["fit_lo"], fit["fit_hi"], 300)
    axL.plot(xx, ds._gaussian(xx, *fit["popt"]) / norm, color="k", lw=2,
             label=f"Gaussian peak fit\nmode = {stats['mode']:.3f} $\\pm$ {stats['mode_err']:.3f} MeV")
    axL.axvspan(fit["fit_lo"], fit["fit_hi"], color="k", alpha=0.06, label="fit window")
    axL.axvline(stats["median"], color="#dd8452", lw=2, ls="-.",
                label=f"median = {stats['median']:.3f} MeV")
    axL.axvline(float(pp.mode_energy_loss), color="#c44e52", lw=2,
                label=f"predicted mode = {float(pp.mode_energy_loss):.2f} MeV")
    axL.axvline(float(pp.mean_energy_loss), color="#55a868", lw=2, ls="--",
                label=f"predicted mean = {float(pp.mean_energy_loss):.2f} MeV")
    axL.set(xlabel="energy loss $\\Delta E$ [MeV]", ylabel="probability density",
            title="Landau energy-loss spectrum")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axL.set_xlim(0, stats["mode"] + 8 * fit["sigma"])

    # --- top-right: outgoing momentum ---
    axR.hist(pc_out, bins=N_BINS, range=(np.percentile(pc_out, 0.5), BEAM_PC),
             color="#8172b3", alpha=0.85, density=True)
    axR.axvline(BEAM_PC, color="0.3", lw=2, ls=":", label=f"incoming {BEAM_PC:.0f} MeV/c")
    axR.set(xlabel="outgoing momentum |p| [MeV/c]", ylabel="probability density",
            title="Momentum after the absorber")
    axR.legend(); axR.grid(alpha=0.3)

    # --- bottom-left: angular deflection theta_x ---
    # plot range = +-5 sigma; overlay Gaussian with the Highland-predicted RMS
    t_lim = 5.0 * theta0
    axT.hist(theta_x, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             color="#937860", alpha=0.85, density=True,
             label=f"empirical (RMS = {theta_x_rms*1e3:.3f} mrad)")
    th = np.linspace(-t_lim, t_lim, 400)
    gauss_theta = np.exp(-0.5 * (th / theta0) ** 2) / (theta0 * np.sqrt(2 * np.pi))
    axT.plot(th, gauss_theta, color="k", lw=2,
             label=f"Highland prediction\n$\\theta_0$ = {theta0*1e3:.3f} mrad")
    axT.set(xlabel="$\\theta_x$ [rad]", ylabel="probability density",
            title="Angular deflection (one plane)")
    axT.legend(fontsize=8); axT.grid(alpha=0.3)

    # --- bottom-right: lateral displacement dx ---
    y_lim = 5.0 * y_rms_pred
    axY.hist(dx, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             color="#da8bc3", alpha=0.85, density=True,
             label=f"empirical (RMS = {dx_rms*1e3:.3f} mm $\\times 10^{{-3}}$)")
    yy = np.linspace(-y_lim, y_lim, 400)
    gauss_y = np.exp(-0.5 * (yy / y_rms_pred) ** 2) / (y_rms_pred * np.sqrt(2 * np.pi))
    axY.plot(yy, gauss_y, color="k", lw=2,
             label=f"PDG prediction\n$x\\theta_0/\\sqrt{{3}}$ = {y_rms_pred*1e3:.3f} $\\mu$m")
    axY.set(xlabel="lateral displacement $\\Delta x$ [mm]", ylabel="probability density",
            title=f"Lateral displacement (corr$(\\theta_x, \\Delta x)$ = {corr_xy:.3f})")
    axY.legend(fontsize=8); axY.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path("absorber_simulation.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png.resolve()}")


if __name__ == "__main__":
    main()
