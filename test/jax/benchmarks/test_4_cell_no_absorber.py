"""Benchmark 4: low-emittance beam through the cooling cell, no absorber.

Table 4 of the cooling code benchmarking note.
"""

import cooling_common as cc
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDz

N_BEAM = 10_000
N_Z = 201  # 4 mm save spacing, fine enough for the aperture check
SEED = 20240702


@pytest.mark.extended
def test_benchmark_4_cooling_cell_no_absorber(artifacts_dir):
    """With no material the transport is symplectic, so emittance growth
    measures integrator error.

    The entry-plane assertions cross-check the beam generator against the
    optics diagnostics: if either drops a factor of the muon mass or beta, the
    recovered emittance and beta function will not match Table 4.
    """
    field = cc.cooling_cell_field()
    mass = float(cc.reference_muon().mass) / u.MeV

    beam = cc.make_beam(
        jr.key(SEED),
        N_BEAM,
        momentum=cc.REF_MOMENTUM,
        beta_perp=107.0 * u.mm,
        eps_perp=2.5e-3 * u.mm,
        sigma_t=0.003532 * u.ns,
        sigma_E=0.3692 * u.MeV,
    )
    zs = jnp.linspace(0.0 * u.mm, cc.CELL_LENGTH, N_Z)

    @jax.jit
    @jax.vmap
    def run(start: MuonStateDz) -> MuonStateDz:
        ys, _ = diffrax_solve(field=field, start=start, cts=zs, rtol=1e-7, atol=1e-9)
        return ys

    track = run(beam)

    # Aperture losses counted on the save grid
    rho = jnp.hypot(track.kin.p.x, track.kin.p.y)
    survived = np.asarray(jnp.all(rho <= cc.APERTURE, axis=-1))
    transmission = float(survived.mean())

    entry = cc.optics(track, survived, 0, mass)
    exit_ = cc.optics(track, survived, -1, mass)

    header = [
        "benchmark 4, Table 4 (low emittance), no absorber",
        f"n_particles={N_BEAM}",
        f"transmission={transmission:.6f}",
        f"rf_phase_deg={cc.RF_PHASE_DEG}",
        f"n_fringe_cells={cc.N_FRINGE}",
        f"aperture_mm={cc.APERTURE / u.mm}",
    ]
    cc.write_cell_artifacts(
        artifacts_dir, "benchmark_4_cell_no_absorber", zs, track,
        survived, mass, header,
    )

    # Generator and diagnostics agree with Table 4.
    assert entry["eps_perp_mm"] == pytest.approx(2.5e-3, rel=0.05)
    assert entry["beta_perp_mm"] == pytest.approx(107.0, rel=0.05)
    assert entry["eps_long_eV_ms"] == pytest.approx(1.304e-3, rel=0.05)
    # Symplectic transport conserves both emittances.
    assert exit_["eps_perp_mm"] == pytest.approx(entry["eps_perp_mm"], rel=0.01)
    assert exit_["eps_long_eV_ms"] == pytest.approx(entry["eps_long_eV_ms"], rel=0.01)
    # A flipped coil polarity shows up here and almost nowhere else.
    assert abs(exit_["L_kin_mm_MeV"]) < 0.05
    # RF sign convention: 20 deg off bunching must accelerate.
    assert exit_["mean_E_MeV"] > entry["mean_E_MeV"]
    # Table 4 sigma_x is 0.376 mm, so the aperture is 217 sigma away.
    assert transmission == pytest.approx(1.0)