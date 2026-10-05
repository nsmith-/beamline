"""Benchmark 4: nominal-emittance beam through the cooling cell with absorbers.

Table 5 of the cooling code benchmarking note.
"""

import cooling_common as cc
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import (
    AbsorberCylinder,
    SumMaterialVolume,
    TransformMaterialVolume,
)
from beamline.jax.coordinates import Transform
from beamline.jax.integrate.stochastic import stochastic_solve
from beamline.jax.kinematics import MuonStateDz

N_BEAM = 10_000
N_Z = 201
SEED = 20240703


@pytest.mark.extended
def test_benchmark_4_cooling_cell_absorber(artifacts_dir):
    """Two LiH absorbers of half the Table 5 thickness, one at each end, so the
    beam crosses one full absorber length per cell.

    Predicted for 10 mm of the PDG LiH entry (rho = 0.82) at 200 MeV/c:
      dE = 1.7145 MeV, X0 = 970.9 mm, x/X0 = 0.0103
      transverse: cooling -0.970%, heating +0.247%, net -0.72%
      longitudinal: sigma_E 11.680 -> 11.688 MeV (+0.065%)
      RF: transit-time factor 0.652, gain 3.687 MeV, net +1.97 MeV per cell
    """
    field = cc.cooling_cell_field()
    mass = float(cc.reference_muon().mass) / u.MeV

    # An absorber at each end
    absorbers = SumMaterialVolume(
        components=[
            TransformMaterialVolume(
                transform=Transform.make_translation(z=zc),
                material=AbsorberCylinder(
                    material=MATERIALS["lithium_hydride_LiH"],
                    radius=cc.APERTURE,
                    length=cc.ABSORBER_HALF,
                ),
            )
            for zc in (
                cc.ABSORBER_HALF / 2,
                cc.CELL_LENGTH - cc.ABSORBER_HALF / 2,
            )
        ]
    )

    beam = cc.make_beam(
        jr.key(SEED),
        N_BEAM,
        momentum=cc.REF_MOMENTUM,
        beta_perp=107.0 * u.mm,
        eps_perp=2.5 * u.mm,
        sigma_t=0.1117 * u.ns,
        sigma_E=11.68 * u.MeV,
    )
    zs = jnp.linspace(0.0 * u.mm, cc.CELL_LENGTH, N_Z)
    keys = jr.split(jr.key(SEED + 1), N_BEAM)

    @jax.jit
    @jax.vmap
    def run(start: MuonStateDz, key) -> MuonStateDz:
        ys, _ = stochastic_solve(field, absorbers, start, zs, key, kick=cc.KICK)
        return ys

    track = run(beam, keys)

    # Exclude both aperture losses and particles stopped by the Landau tail
    rho = jnp.hypot(track.kin.p.x, track.kin.p.y)
    finite = jnp.all(jnp.isfinite(track.kin.t.z), axis=-1)
    survived = np.asarray(jnp.all(rho <= cc.APERTURE, axis=-1) & finite)
    transmission = float(survived.mean())

    entry = cc.optics(track, survived, 0, mass)
    exit_ = cc.optics(track, survived, -1, mass)

    lih = MATERIALS["lithium_hydride_LiH"]
    header = [
        "benchmark 4, Table 5 (nominal emittance), 5 mm LiH at each end",
        f"material={lih.name}",
        f"density_g_cm3={lih.density / (u.g / u.cm3):.5f}",
        "NOTE density is the PDG tabulated 0.82 g/cm3; benchmark PDF Table 1 "
        "specifies 0.69 for LiH",
        f"absorber_half_thickness_mm={cc.ABSORBER_HALF / u.mm}",
        f"n_particles={N_BEAM}",
        f"transmission={transmission:.6f}",
        f"rf_phase_deg={cc.RF_PHASE_DEG}",
        f"eps_perp_in={entry['eps_perp_mm']:.6f}",
        f"eps_perp_out={exit_['eps_perp_mm']:.6f}",
        f"eps_long_in={entry['eps_long_eV_ms']:.6f}",
        f"eps_long_out={exit_['eps_long_eV_ms']:.6f}",
    ]
    cc.write_cell_artifacts(
        artifacts_dir, "benchmark_4_cell_absorber", zs, track,
        survived, mass, header,
    )

    # Table 5 sigma_x is 11.9 mm, so the aperture is only 6.9 sigma away and
    # some loss is expected; anything drastic means the optics are wrong.
    assert transmission > 0.5, f"transmission {transmission:.3f} implausibly low"
    assert exit_["eps_perp_mm"] < entry["eps_perp_mm"], (
        "transverse emittance did not cool; check absorber placement and "
        "RF phase sign"
    )