"""
Simulate a muon beam passing through TWO solid SiO2 disks placed back-to-back
along z.

Geometry: the two disks are touching (no field-free gap between them).
A particle:
  1) enters disk 1, where energy loss and multiple scattering are applied
     in place (CylindricalAbsorber.apply does NOT propagate longitudinally)
  2) drifts through disk 2's thickness using its post-disk-1 direction,
     accumulating the transverse position offset that disk-1's exit angle
     would produce as the particle traverses the second disk
  3) enters disk 2, where energy loss and multiple scattering are applied
     in place

Without step 2, the simulation would treat the two disks as two thin
scatterers stacked at the same z-coordinate, missing the contribution of
disk-1's exit angle to the final transverse position. Step 2 restores
this contribution and makes the simulation faithful to a touching-disks
geometry.

This script plots four observables:

  1) Landau energy-loss spectrum (one disk vs both)
  2) Outgoing momentum magnitude
  3) Transverse angular deflection theta_x
  4) Transverse lateral displacement dx

For the scattering panels, two reference theory lines are shown:
  - The two-disk-touching theory derived for this exact geometry,
    including disk-1's angle propagating through disk-2's thickness:
        sigma^2(dx) = 8/3 * x^2 * theta_0^2
    where x is the per-disk thickness. RMS = sqrt(8/3) * x * theta_0
    ~= 112.4 um for 5+5 mm SiO2 at 200 MeV/c
  - The single-10mm Highland prediction for the same total thickness,
    which serves as a "continuous slab" reference. The two-disk-touching
    and single-thick-slab numbers differ at the ~3% level because
    Highland's log term is not linear in x; this discretization gap is
    the expected difference between treating the material as two thin
    scatterers vs one continuous slab.

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
import equinox as eqx

from beamline.jax.absorber.absorber import CylindricalAbsorber
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent

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


def drift_through_thickness(state, thickness):
    """Advance the particle's transverse position by theta * thickness.

    For a particle with momentum (px, py, pz) traversing a longitudinal
    distance `thickness`, the transverse position changes by
        dx = (px / pz) * thickness ~= theta_x * thickness  (small angle)
        dy = (py / pz) * thickness ~= theta_y * thickness

    This is the inter-disk longitudinal-propagation step that
    CylindricalAbsorber.apply doesn't perform. We use it to drift the
    particle across disk 2's thickness between the two apply() calls,
    so disk-1's exit angle contributes to the final transverse
    position the same way it would in a continuous-slab simulation.
    """
    px, py, pz = state.kin.t.x, state.kin.t.y, state.kin.t.z
    dx = px * thickness / pz
    dy = py * thickness / pz

    x_pos, y_pos = state.kin.p.x, state.kin.p.y
    new_position = Cartesian4.make(
        x=x_pos + dx,
        y=y_pos + dy,
        z=state.kin.p.z,
        ct=state.kin.p.ct,
    )
    new_kin = Tangent(p=new_position, t=state.kin.t)
    return eqx.tree_at(lambda s: s.kin, state, new_kin)


def main():
    mat = MATERIALS[MATERIAL]
    disk1 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK1_LEN * u.mm)
    disk2 = CylindricalAbsorber(material=mat, radius=RADIUS * u.mm, length=DISK2_LEN * u.mm)

    def through_two_disks(state, key):
        # disk 1: in-place energy loss + scattering
        state, key = disk1.apply(state, key)
        # propagate the post-disk-1 direction across disk-2's thickness;
        # this is the contribution missing from a naive thin-scatterer chain
        state = drift_through_thickness(state, disk2.length)
        # disk 2: in-place energy loss + scattering on the drifted state
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

    # Single-disk observables (for comparison; just disk 1 alone)
    out1, _ = jax.jit(jax.vmap(disk1.apply))(beam, keys)
    dE_one = np.asarray(beam.kin.t.ct - out1.kin.t.ct)
    theta_x_one = np.arctan2(np.asarray(out1.kin.t.x), np.asarray(out1.kin.t.z))
    dx_one = np.asarray(out1.kin.p.x - beam.kin.p.x)

    # Theory predictions
    probe = make_muon(BEAM_PC)
    pp1 = mat.straggling_params(probe, disk1.length)
    theta0_one  = float(mat.highland_theta0(probe, disk1.length))
    theta0_10mm = float(mat.highland_theta0(probe, disk1.length + disk2.length))
    y_rms_one_pred = float(disk1.length) * theta0_one / np.sqrt(3.0)

    # Two-disk-touching theory: includes disk-1 in-disk offset, disk-2
    # in-disk offset, disk-1 angle propagated through disk 2, and the
    # cov(y_1, theta_1) cross-term (rho = sqrt(3)/2).
    #   Var(dx) = 2 * (x*theta_0)^2 / 3      (in-disk offsets)
    #           + (theta_0)^2 * x^2          (disk-1 angle drift through disk 2)
    #           + 2 * rho * sigma(y) * sigma(theta) * x  (cross-term)
    # For x_1 = x_2 = x this evaluates to 8/3 * x^2 * theta_0^2.
    x1, x2 = float(disk1.length), float(disk2.length)
    var_two_disk = (
        x1**2 * theta0_one**2 / 3.0      # disk-1 in-disk offset
        + x2**2 * theta0_one**2 / 3.0    # disk-2 in-disk offset
        + theta0_one**2 * x2**2          # disk-1 angle through disk 2
        + 2 * (np.sqrt(3) / 2)           # rho
          * (x1 * theta0_one / np.sqrt(3))   # sigma(y_1)
          * theta0_one                        # sigma(theta_1)
          * x2                                # propagation distance
    )
    y_rms_two_touching = float(np.sqrt(var_two_disk))

    # Single-10mm-disk reference for the same total thickness.
    y_rms_10mm = (x1 + x2) * theta0_10mm / np.sqrt(3.0)

    # Two-disk theta_0 (variances add since angle kicks are independent).
    theta0_two_disk = np.sqrt(2.0) * theta0_one

    # --- precise mode/median/mean with uncertainties ---
    hr1 = (0.0, float(np.percentile(dE_one,   99.5)))
    hr2 = (0.0, float(np.percentile(dE_total, 99.5)))
    stats1 = ds.summarize(dE_one,   name=f"one {DISK1_LEN:.0f} mm disk",
                          n_boot=N_BOOT, bins=N_BINS, hist_range=hr1, seed=SEED)
    stats2 = ds.summarize(dE_total, name="two disks (summed loss)",
                          n_boot=N_BOOT, bins=N_BINS, hist_range=hr2, seed=SEED)

    print(f"Beam            : {BEAM_PC:.0f} MeV/c muons, N = {N_PARTICLES:,}")
    print(f"Disks           : {DISK1_LEN:.0f} mm + {DISK2_LEN:.0f} mm {mat.name} (touching)")
    print(f"momentum {BEAM_PC:.0f} -> {pc_out.mean():.2f} MeV/c")
    print("-" * 56)
    print("ENERGY LOSS:")
    print(f"  one disk  : fitted mode {stats1['mode']:.3f} +/- {stats1['mode_err']:.3f} MeV  "
          f"(predicted {float(pp1.mode_energy_loss):.3f})")
    print(f"  two disks : fitted mode {stats2['mode']:.3f} +/- {stats2['mode_err']:.3f} MeV")
    print(f"              median {stats2['median']:.3f} +/- {stats2['median_err']:.3f}, "
          f"mean {stats2['mean']:.3f} +/- {stats2['mean_err']:.3f} MeV")
    print("-" * 56)
    print("SCATTERING:")
    print(f"  one disk      : empirical theta_x RMS = {np.std(theta_x_one)*1e3:.4f} mrad")
    print(f"                  predicted theta_0     = {theta0_one*1e3:.4f} mrad")
    print(f"                  empirical dx RMS      = {np.std(dx_one)*1e3:.4f} um")
    print(f"                  predicted dx RMS      = {y_rms_one_pred*1e3:.4f} um")
    print(f"  two disks     : empirical theta_x RMS = {np.std(theta_x_two)*1e3:.4f} mrad")
    print(f"                  predicted theta_0     = {theta0_two_disk*1e3:.4f} mrad  (sqrt(2) * single-5mm)")
    print(f"                  single-10mm Highland  = {theta0_10mm*1e3:.4f} mrad  (continuous-slab ref)")
    print(f"                  empirical dx RMS      = {np.std(dx_two)*1e3:.4f} um")
    print(f"                  two-disk-touching     = {y_rms_two_touching*1e3:.4f} um  (theory for this geometry)")
    print(f"                  single-10mm Highland  = {y_rms_10mm*1e3:.4f} um  (continuous-slab ref)")

    # ---- plotting -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axL, axR), (axT, axY) = axes
    fig.suptitle(f"{BEAM_PC:.0f} MeV/c muons through two {DISK1_LEN:.0f} mm {mat.name} disks (touching)",
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
    # Two reference lines: sqrt(2) * theta_0(5mm) for two-disk stacking,
    # and theta_0(10mm) for the continuous-slab reference.
    t_lim = 5.0 * theta0_two_disk
    axT.hist(theta_x_one, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(theta_x_one)*1e3:.3f} mrad)")
    axT.hist(theta_x_two, bins=N_BINS_MCS, range=(-t_lim, t_lim),
             histtype="step", lw=2, color="#937860", density=True,
             label=f"two disks (RMS = {np.std(theta_x_two)*1e3:.3f} mrad)")
    th = np.linspace(-t_lim, t_lim, 400)
    g_one    = np.exp(-0.5 * (th / theta0_one)**2)     / (theta0_one    * np.sqrt(2 * np.pi))
    g_two    = np.exp(-0.5 * (th / theta0_two_disk)**2) / (theta0_two_disk * np.sqrt(2 * np.pi))
    g_10mm   = np.exp(-0.5 * (th / theta0_10mm)**2)    / (theta0_10mm   * np.sqrt(2 * np.pi))
    axT.plot(th, g_one, color="0.4", lw=1.5, ls=":",
             label=f"one-disk Highland ($\\theta_0$ = {theta0_one*1e3:.3f} mrad)")
    axT.plot(th, g_two, color="k", lw=2,
             label=f"two-disk theory ($\\sqrt{{2}}\\,\\theta_0$ = {theta0_two_disk*1e3:.3f} mrad)")
    axT.plot(th, g_10mm, color="#c44e52", lw=1.5, ls="--",
             label=f"single-10mm Highland ($\\theta_0$ = {theta0_10mm*1e3:.3f} mrad)")
    axT.set(xlabel="$\\theta_x$ [rad]", ylabel="probability density",
            title="Angular deflection (one plane)")
    axT.legend(fontsize=8); axT.grid(alpha=0.3)

    # --- bottom-right: lateral displacement dx ---
    # Two reference lines: two-disk-touching theory (the right comparison for
    # this geometry) and single-10mm Highland (continuous-slab limit).
    y_lim = 5.0 * y_rms_two_touching
    axY.hist(dx_one, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             color="#c7c7c7", alpha=0.85, density=True,
             label=f"one disk (RMS = {np.std(dx_one)*1e3:.3f} mm $\\times 10^{{-3}}$)")
    axY.hist(dx_two, bins=N_BINS_MCS, range=(-y_lim, y_lim),
             histtype="step", lw=2, color="#da8bc3", density=True,
             label=f"two disks (RMS = {np.std(dx_two)*1e3:.3f} mm $\\times 10^{{-3}}$)")
    yy = np.linspace(-y_lim, y_lim, 400)
    g_y_one      = np.exp(-0.5 * (yy / y_rms_one_pred)**2)     / (y_rms_one_pred     * np.sqrt(2 * np.pi))
    g_y_touching = np.exp(-0.5 * (yy / y_rms_two_touching)**2) / (y_rms_two_touching * np.sqrt(2 * np.pi))
    g_y_10mm     = np.exp(-0.5 * (yy / y_rms_10mm)**2)         / (y_rms_10mm         * np.sqrt(2 * np.pi))
    axY.plot(yy, g_y_one, color="0.4", lw=1.5, ls=":",
             label=f"one-disk PDG ($x\\theta_0/\\sqrt{{3}}$ = {y_rms_one_pred*1e3:.3f} $\\mu$m)")
    axY.plot(yy, g_y_touching, color="k", lw=2,
             label=f"two-disk touching theory ({y_rms_two_touching*1e3:.3f} $\\mu$m)")
    axY.plot(yy, g_y_10mm, color="#c44e52", lw=1.5, ls="--",
             label=f"single-10mm Highland ({y_rms_10mm*1e3:.3f} $\\mu$m)")
    axY.set(xlabel="lateral displacement $\\Delta x$ [mm]", ylabel="probability density",
            title="Lateral displacement")
    axY.legend(fontsize=8); axY.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path("two_disks_simulation.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png.resolve()}")


if __name__ == "__main__":
    main()
