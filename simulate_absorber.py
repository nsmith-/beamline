"""
Simulate a muon beam passing through the SiO2 (fused quartz) cylinder.

Each muon arrives at the absorber face and receives a stochastic energy-loss
kick sampled from the Landau distribution (sample_energy_loss -> RANLAN), which
is applied as a direction-preserving reduction of its momentum. We run a whole
beam at once with jax.vmap, then look at the energy-loss spectrum.

Run from the repo root:
    python simulate_absorber.py
(The script adds ./src to the path if `beamline` isn't already importable.)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `beamline` importable when run from the repo root without installing.
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

# ---------------------------------------------------------------- configuration
MATERIAL    = "silicon_dioxide_SiO2"
BEAM_PC     = 200.0      # beam momentum [MeV/c]
RADIUS      = 100.0      # cylinder radius [mm]
LENGTH      = 10.0       # cylinder thickness [mm]  (= dE/dx path length)
N_PARTICLES = 200_000
SEED        = 42


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
    E_in = float(probe.kin.t.ct)
    pc_in = float(jnp.sqrt(jnp.sum(probe.kin.t.coords[:3] ** 2)))

    # ----- run the beam: one PRNG key per particle, applied with vmap+jit -----
    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    run = jax.jit(jax.vmap(absorber.apply))
    out, _ = run(beam, keys)

    dE = np.asarray(beam.kin.t.ct - out.kin.t.ct)          # energy lost [MeV]
    pc_out = np.asarray(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))

    # empirical peak of the spectrum
    h, edges = np.histogram(dE, bins=600)
    mode_emp = 0.5 * (edges[h.argmax()] + edges[h.argmax() + 1])

    print(f"Material              : {absorber.material.name}")
    print(f"Beam                  : {BEAM_PC:.0f} MeV/c muons, N = {N_PARTICLES:,}")
    print(f"Absorber              : {LENGTH:.0f} mm thick, R = {RADIUS:.0f} mm")
    print(f"E_in                  : {E_in:.3f} MeV   (KE = {E_in - float(probe.mass):.3f} MeV)")
    print("-" * 56)
    print(f"predicted xi          : {float(pp.xi):.4f} MeV")
    print(f"predicted mode (Landau): {float(pp.mode_energy_loss):.4f} MeV")
    print(f"empirical  mode        : {mode_emp:.4f} MeV   <-- should match the line above")
    print(f"predicted mean (Bethe) : {float(pp.mean_energy_loss):.4f} MeV")
    print(f"empirical  median      : {np.median(dE):.4f} MeV")
    print(f"empirical  mean        : {dE.mean():.4f} MeV   (> mode: Landau's heavy tail)")
    print(f"kappa                 : {float(pp.kappa):.4f}  (Landau valid for kappa << 1)")
    print(f"momentum  {pc_in:.1f} -> {pc_out.mean():.2f} MeV/c (mean)")

    # ----------------------------------------------------------------- plotting
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{BEAM_PC:.0f} MeV/c muon beam through {LENGTH:.0f} mm {absorber.material.name}",
        fontsize=13, fontweight="bold",
    )

    hi = np.percentile(dE, 99.5)                 # clip the long tail for display
    axL.hist(dE, bins=400, range=(0, hi), color="#4c72b0", alpha=0.85, density=True)
    axL.axvline(float(pp.mode_energy_loss), color="#c44e52", lw=2,
                label=f"predicted mode = {float(pp.mode_energy_loss):.2f} MeV")
    axL.axvline(float(pp.mean_energy_loss), color="#55a868", lw=2, ls="--",
                label=f"predicted mean = {float(pp.mean_energy_loss):.2f} MeV")
    axL.set(xlabel="energy loss $\\Delta E$ [MeV]", ylabel="probability density",
            title="Landau energy-loss spectrum")
    axL.legend(); axL.grid(alpha=0.3)
    axL.text(0.97, 0.55, "long high-loss tail\n(extends past plot)",
             transform=axL.transAxes, ha="right", fontsize=9, color="0.4")

    axR.hist(pc_out, bins=400, range=(np.percentile(pc_out, 0.5), BEAM_PC),
             color="#8172b3", alpha=0.85, density=True)
    axR.axvline(BEAM_PC, color="0.3", lw=2, ls=":", label=f"incoming {BEAM_PC:.0f} MeV/c")
    axR.set(xlabel="outgoing momentum |p| [MeV/c]", ylabel="probability density",
            title="Momentum after the absorber")
    axR.legend(); axR.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = Path("absorber_simulation.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png.resolve()}")


if __name__ == "__main__":
    main()
