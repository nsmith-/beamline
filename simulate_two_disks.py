"""
Simulate a muon beam passing through TWO solid SiO2 disks spaced along z.
 
Each disk is just a short CylindricalAbsorber. The beam runs along +z, normal
to both faces, so each disk's `length` IS the true path length. A particle
crosses disk 1 (Landau sampling #1), drifts the field-free gap (no loss), then
crosses disk 2 (Landau sampling #2), applied to the already-degraded state,
so the second disk correctly sees slightly lower momentum.
 
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
 
# ---------------------------------------------------------------- configuration
MATERIAL    = "silicon_dioxide_SiO2"
BEAM_PC     = 200.0      # beam momentum [MeV/c]
RADIUS      = 100.0      # disk radius [mm]
DISK1_LEN   = 5.0        # disk 1 thickness [mm]
DISK2_LEN   = 5.0        # disk 2 thickness [mm]
N_PARTICLES = 200_000
SEED        = 7
 
 
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
        state, key = disk1.apply(state, key)   # sampling #1
        state, key = disk2.apply(state, key)   # sampling #2 (sees degraded state)
        return state
 
    keys = jax.random.split(jax.random.key(SEED), N_PARTICLES)
    beam = jax.vmap(make_muon)(jnp.full(N_PARTICLES, BEAM_PC))
    out = jax.jit(jax.vmap(through_two_disks))(beam, keys)
 
    dE_total = np.asarray(beam.kin.t.ct - out.kin.t.ct)
 
    # For context: the loss from a single disk on its own.
    out1, _ = jax.jit(jax.vmap(disk1.apply))(beam, keys)
    dE_one_disk = np.asarray(beam.kin.t.ct - out1.kin.t.ct)
 
    def mode(d):
        h, e = np.histogram(d, bins=400, range=(np.percentile(d, 1), np.percentile(d, 90)))
        return 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
 
    m1 = float(disk1.material.straggling_params(make_muon(BEAM_PC), disk1.length).mode_energy_loss)
    print(f"Beam            : {BEAM_PC:.0f} MeV/c muons, N = {N_PARTICLES:,}")
    print(f"Disks           : {DISK1_LEN:.0f} mm + {DISK2_LEN:.0f} mm {mat.name}")
    print(f"single {DISK1_LEN:.0f}mm disk : mode {mode(dE_one_disk):.3f}  (predicted {m1:.3f}) MeV")
    print(f"TWO disks (sum)  : mode {mode(dE_total):.3f}  median {np.median(dE_total):.3f} MeV")
    print(f"momentum {BEAM_PC:.0f} -> {float(jnp.mean(jnp.sqrt(jnp.sum(out.kin.t.coords[:, :3] ** 2, axis=1)))):.2f} MeV/c")
 
    # ----------------------------------------------------------------- plotting
    fig, axA = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"{BEAM_PC:.0f} MeV/c muons through two {DISK1_LEN:.0f} mm {mat.name} disks",
                 fontsize=13, fontweight="bold")
 
    hiA = np.percentile(dE_total, 99.5)
    axA.hist(dE_one_disk, bins=300, range=(0, hiA), density=True,
             color="#c7c7c7", alpha=0.9, label=f"one {DISK1_LEN:.0f} mm disk (1 sampling)")
    axA.hist(dE_total, bins=300, range=(0, hiA), density=True,
             histtype="step", lw=2, color="#4c72b0", label="both disks (2 samplings, summed)")
    axA.set(xlabel="energy loss $\\Delta E$ [MeV]", ylabel="probability density",
            title="One disk vs the sum of two")
    axA.legend(); axA.grid(alpha=0.3)
 
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = Path("two_disks_simulation.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png.resolve()}")
 
 
if __name__ == "__main__":
    main()
