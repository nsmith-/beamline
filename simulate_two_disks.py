"""
Simulate a muon beam passing through TWO solid SiO2 disks spaced along z.

Each disk is just a short CylindricalAbsorber. The beam runs along +z, normal
to both faces, so each disk's `length` IS the true path length. A particle
crosses disk 1 (energy-loss + scattering #1), drifts the field-free gap (no
loss, no scattering), then crosses disk 2 (energy-loss + scattering #2),
applied to the already-degraded state, so the second disk correctly sees
slightly lower momentum.

This version plots both energy loss and scattering observables, comparing
the one-disk and two-disk distributions:

  1. The energy-loss spectrum                       (Landau)
  2. The outgoing momentum magnitude                (consequence of dE)
  3. The transverse angular deflection theta_x      (Gaussian, RMS grows
                                                     as sqrt(2) for two disks)
  4. The transverse lateral displacement dx         (Gaussian)

Theory notes for the two-disk case:
  - Angular variances add: var(theta_x_total) = var(disk1) + var(disk2),
    so for two identical disks the total RMS is sqrt(2) * theta_0(one disk).
  - Lateral displacement for two disks is NOT just sqrt(2) * (one disk),
    because the angle from disk 1 propagates as a position offset at
    disk 2 (drift between disks contributes geometrically). The theory
    line shown is "Highland applied to the combined 10 mm thickness",
    which is the correct prediction if the two disks are touching;
    if there's a gap, the empirical y RMS will be larger.

Run from the repo root:
    python simulate_two_disks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

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

# configuration
MATERIAL    = "silicon_dioxide_SiO2"
BEAM_PC     = 200.0      # MeV/c
RADIUS      = 100.0      # mm
DISK1_LEN   = 5.0        # mm
DISK2_LEN   = 5.0        # mm
N_PARTICLES = 1000000
SEED        = 7
N_BOOT      = 500
N_BINS      = 2000
N_BINS_MCS  = 200


def make_muon(pc_MeV):
    return MuonStateDz.make(
        position=Cartesian4.make(z=0.0),
        momentum=Cartesian3.make(z=pc_MeV * u.MeV),
        q=1,
    )


def main():
    mat = MATERIALS[MATERIAL]
    disk1 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK1_LEN * u.mm)
    disk2 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK2_LEN * u.mm)

    def through_two_disks(state, key):
        state, key = disk1.apply(state, key)
        state, key = disk2.apply(state, key)
        return state

    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    out = jax.jit(jax.vmap(through_two_disks))(beam, keys)

    # Two-disk observables
    dE_total = np.asarray(beam.kin.t.ct - out.kin.t.ct)
    pc_out = np.asarray(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))
    theta_x_two = np.arctan2(np.asarray(out.kin.t.x), np.asarray(out.kin.t.z))
    dx_two = np.asarray(out.kin.p.x - beam.kin.p.x)

    # Single-disk observables (for comparison)
    out1, _ = jax.jit(jax.vmap(disk1.apply))(beam, keys)
    dE_one = np.asarray(beam.kin.t.ct - out1.kin.t.ct)
    theta_x_one = np.arctan2(np.asarray(out1.kin.t.x), np.asarray(out1.kin.t.z))
    dx_one = np.asarray(out1.kin.p.x - beam.kin.p.x)

    # Theory predictions
    probe = make_muon(BEAM_PC)
    pp1 = mat.straggling_params(probe, disk1.length)
    theta0_one = float(mat.highland_theta0(probe, disk1.length))
    y_rms_one_pred = float(disk1.length) * theta0_one / np.sqrt(3.0)

    # For two touching disks the right prediction is Highland evaluated
    # at the combined thickness (10 mm), not sqrt(2) * one-disk theta_0:
    combined_length = disk1.length + disk2.length
    theta0_two = float(mat.highland_theta0(probe, combined_length))
    y_rms_two_pred = float(combined_length) * theta0_two / np.sqrt(3.0)

    # --- precise mode/median/mean with uncertainties ---
    hr1 = (0.0, float(np.percentile(dE_one,   99.5)))
    hr2 = (0.0, float(np.percentile(dE_total, 99.5)))
    stats1 = ds.summarize(dE_one,   name=f"one {DISK1_LEN:.0f} mm disk",
                          n_boot=N_BOOT, bins=N_BINS, hist_range=hr1, seed=SEED)
    stats2 = ds.summarize(dE_total, name="two disks (summed loss)",
                          n_boot=N_BOOT, bins=N_BINS, hist_range=hr2, seed=SEED)

    print(f"Beam            : {BEAM_PC:.0f} MeV/c muons, N = {N_PARTICLES:,}")
    print(f"Disks           : {DISK1_LEN:.0f} mm + {DISK2_LEN:.0f} mm {mat.name}")
    print(f"momentum {BEAM_PC:.0f} -> {pc_out.mean():.2f} MeV/c")
    print("-" * 56)
    print("ENERGY LOSS:")
    print(f"  one disk    : fitted mode {stats1['mode']:.3f} +/- {stats1['mode_err']:.3f} MeV  "
          f"(predicted {float(pp1.mode_energy_loss):.3f})")
    print(f"  two disks   : fitted mode {stats2['mode']:.3f} +/- {stats2['mode_err']:.3f} MeV")
    print(f"  two disks   : median {stats2['median']:.3f} +/- {stats2['median_err']:.3f}, "
          f"mean {stats2['mean']:.3f} +/- {stats2['mean_err']:.3f} MeV")
    print("-" * 56)
    print("SCATTERING:")
    print(f"  one disk    : predicted theta_0 = {theta0_one*1e3:.4f} mrad")
    print(f"                empirical theta_x RMS = {np.std(theta_x_one)*1e3:.4f} mrad")
    print(f"                empirical dx RMS      = {np.std(dx_one)*1e3:.4f} um")
    print(f"                predicted dx RMS      = {y_rms_one_pred*1e3:.4f} um")
    print(f"  two disks   : predicted theta_0 = {theta0_two*1e3:.4f} mrad  "
          f"(Highland at combined {combined_length:.0f} mm)")
    print(f"                empirical theta_x RMS = {np.std(theta_x_two)*1e3:.4f} mrad")
    print(f"                empirical dx RMS      = {np.std(dx_two)*1e3:.4f} um")
    print(f"                predicted dx RMS      = {y_rms_two_pred*1e3:.4f} um")

    # ---- plotting -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axL, axR), (axT, axY) = axes
    fig.suptitle(f"{BEAM_PC:.0f} MeV/c muons through two {DISK1_LEN:.0f} mm {mat.name} disks",
                 fontsize=13, fontweight="bold")

    # --- top-left: energy-loss spectrum ---
    hiA = hr2[1]
    axL.hist(dE_one, bins=N_BINS, range=(0, hiA), density=True,
             color="#c7c7c7", alpha=0.9, label=f"one {DISK1_LEN:.0f} mm disk")
    axL.hist(dE_total, bins=N_BINS, range=(0, hiA), density=True,
             histtype="step", lw=2, color="#4c72b0", label="both disks (summed)")
    fit2 = stats2["_fit"]
    bin_width = hr2[1] / N_BINS
    norm = len(dE_total) * bin_width
    xx = np.linspace(fit2["fit_lo"], fit2["fit_hi"], 300)
    axL.plot(xx, ds._gaussian(xx, *fit2["popt"]) / norm, color="k", lw=2,
             label=f"two-disk peak fit\nmode = {stats2['mode']:.3f} $\\pm$ {stats2['mode_err']:.3f} MeV")
    axL.axvline(stats2["median"], color="#dd8452", lw=2, ls="-.",
                label=f"two-disk median = {stats2['median']:.3f} MeV")
    axL.set(xlabel="energy loss $\\Delta E$ [MeV]", ylabel="probability density",
            title="Landau energy-loss spectrum")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axL.set_xlim(0, stats2["mode"] + 8 * fit2["sigma"])

    # --- top-right: outgoing momentum ---
    axR.hist(pc_out, bins=N_BINS, range=(np.percentile(pc_out, 0.5), BEAM_PC),
             color="#8172b3", alpha=0.85, density=True)
    axR.axvline(BEAM_PC, color="0.3", lw=2, ls=":", label=f"incoming {BEAM_PC:.0f} MeV/c")
    axR.set(xlabel="outgoing momentum |p| [MeV/c]", ylabel="probability density",
            title="Momentum after both disks")
    axR.legend(); axR.grid(alpha=0.3)

    # --- bottom-left: angular deflection theta_x ---
    t_lim = 5.0 * theta0_two
    axT.hist(theta_x_one, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(theta_x_one)*1e3:.3f} mrad)")
    axT.hist(theta_x_two, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             histtype="step", lw=2, color="#937860", density=True,
             label=f"two disks (RMS = {np.std(theta_x_two)*1e3:.3f} mrad)")
    th = np.linspace(-t_lim, t_lim, 400)
    g_one = np.exp(-0.5 * (th / theta0_one) ** 2) / (theta0_one * np.sqrt(2 * np.pi))
    g_two = np.exp(-0.5 * (th / theta0_two) ** 2) / (theta0_two * np.sqrt(2 * np.pi))
    axT.plot(th, g_one, color="0.4", lw=1.5, ls="--",
             label=f"one-disk Highland ($\\theta_0$ = {theta0_one*1e3:.3f} mrad)")
    axT.plot(th, g_two, color="k", lw=2,
             label=f"two-disk Highland ($\\theta_0$ = {theta0_two*1e3:.3f} mrad)")
    axT.set(xlabel="$\\theta_x$ [rad]", ylabel="probability density",
            title="Angular deflection (one plane)")
    axT.legend(fontsize=8); axT.grid(alpha=0.3)

    # --- bottom-right: lateral displacement dx ---
    # Two-disk dx theory: each disk contributes BOTH an in-disk offset (y_i)
    # and an out-of-disk drift from the prior disk's angular kick. With no
    # gap between disks, the per-disk drift contribution is theta_1 * x_2.
    # The full second-moment calculation, accounting for the sqrt(3)/2
    # correlation between theta_1 and y_1, gives:
    #
    #   Var(dx_two) = 2 * (x_1 * theta_0)^2 / 3                  # in-disk offsets
    #               + (theta_0)^2 * x_2^2                        # disk-1 angle drift through disk 2
    #               + 2 * (sqrt(3)/2) * (x_1 * theta_0 / sqrt(3)) * theta_0 * x_2
    #                                                            # cross-term between disk-1 (y_1, theta_1)
    #
    # For two touching identical disks of thickness x with single-disk
    # theta_0_single, this simplifies. We use the closed form below.
    x1 = float(disk1.length)
    x2 = float(disk2.length)
    # exact two-disk-no-drift variance (touching disks):
    var_two = (x1**2 * theta0_one**2 / 3.0      # disk-1 in-disk offset
               + x2**2 * theta0_one**2 / 3.0    # disk-2 in-disk offset
               + theta0_one**2 * x2**2          # disk-1 angle propagated through disk 2
               + 2 * (np.sqrt(3)/2) * (x1 * theta0_one / np.sqrt(3)) * theta0_one * x2
              )                                  # correlation cross-term
    y_rms_two_correct = float(np.sqrt(var_two))

    y_lim = 5.0 * y_rms_two_correct
    axY.hist(dx_one, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(dx_one)*1e3:.3f} mm $\\times 10^{{-3}}$)")
    axY.hist(dx_two, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             histtype="step", lw=2, color="#da8bc3", density=True,
             label=f"two disks (RMS = {np.std(dx_two)*1e3:.3f} mm $\\times 10^{{-3}}$)")
    yy = np.linspace(-y_lim, y_lim, 400)
    g_y_one = np.exp(-0.5 * (yy / y_rms_one_pred) ** 2) / (y_rms_one_pred * np.sqrt(2 * np.pi))
    g_y_two = np.exp(-0.5 * (yy / y_rms_two_correct) ** 2) / (y_rms_two_correct * np.sqrt(2 * np.pi))
    axY.plot(yy, g_y_one, color="0.4", lw=1.5, ls="--",
             label=f"one-disk PDG ($x\\theta_0/\\sqrt{{3}}$ = {y_rms_one_pred*1e3:.3f} $\\mu$m)")
    axY.plot(yy, g_y_two, color="k", lw=2,
             label=f"two-disk theory ({y_rms_two_correct*1e3:.3f} $\\mu$m)")
    axY.set(xlabel="lateral displacement $\\Delta x$ [mm]", ylabel="probability density",
            title="Lateral displacement")
    axY.legend(fontsize=8); axY.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path("two_disks_simulation.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png.resolve()}")


if __name__ == "__main__":
    main()
