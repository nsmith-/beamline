"""Cooling code benchmark test 4 (low emittance, no absorber)

As described in:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf

Cell layout (Figure 1, Table 4), in cell-local coordinates with the cell starting at
z=0 (where the absorbers would sit, at the cell boundaries):
- a pair of oppositely polarised coils centered at z=100.7 mm and z=L-100.7 mm
- three TM010 pillbox cavities centered at L/2 and L/2 +- 188.6 mm, phased 180 degrees
  relative to their neighbours
All cells are identical, so the coils either side of a cell boundary have opposite
polarity and the field flips at the absorber. This is the reading under which the
lattice is stable at 200 MeV/c with a periodic beta of ~107 mm at the cell boundary,
matching the beam of Table 4 (alternating the cell polarity instead puts 200 MeV/c
in a stop band).
"""

import importlib.util
import json
from pathlib import Path

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import (
    EMTensorField,
    StackedField,
    SumField,
    TransformEMField,
)
from beamline.jax.geometry import BeamPipe
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity
from beamline.jax.util.beam import beam_moments
from beamline.jax.util.state_io import write_states_csv

# Table 4: cooling cell
CELL_LENGTH = 800.0 * u.mm
BEAM_PIPE_RADIUS = 81.6 * u.mm
COIL_Z = 100.7 * u.mm
RF_PITCH = 188.6 * u.mm
RF_PHASE_OFFSET = 20.0 * u.deg
# Table 2: coil
COIL_RIN = 250.0 * u.mm
COIL_ROUT = 419.3 * u.mm
COIL_LENGTH = 140.0 * u.mm
COIL_JPHI = 500.0 * u.A / u.mm**2
# Table 3: RF cavity
RF_FREQUENCY = 704.0 * u.MHz
RF_E0 = 30.0 * u.MV / u.m
RF_LENGTH = 183.6 * u.mm
# Table 4: beam
P_REF = 200.0 * u.MeV
EMITTANCE_T = 2.5e-3 * u.mm
BETA_T = 107.0 * u.mm
SIGMA_T = 0.003532 * u.ns
SIGMA_E = 0.3692 * u.MeV

REF_MUON = MuonStateDz.make(
    position=Cartesian4.make(),
    momentum=Cartesian3.make(z=P_REF),
    q=1,
)
BETA_REF = REF_MUON.beta()


def setup_cell() -> EMTensorField:
    """Sum contributions from two oppositely polarised solenoids and 3 rf cavities

    The reference particle enters the cell at z=0, ct=0 with constant velocity. The
    central cavity is phased so that the reference particle sees the field
    RF_PHASE_OFFSET past the bunching zero crossing at the cavity center. For a beam
    below transition (as in a solenoid channel without dispersion), bunching means
    late particles gain energy, i.e. the rising zero crossing of Ez, and a positive
    offset accelerates the reference particle.
    """

    def coil(z: float, jphi: float) -> EMTensorField:
        return TransformEMField(
            transform=Transform.make_translation(z=z),
            field=ThickSolenoid(
                Rin=COIL_RIN,
                Rout=COIL_ROUT,
                jphi=jphi,
                L=COIL_LENGTH,
            ),
        )

    omega = 2 * jnp.pi * RF_FREQUENCY
    ct_center = (CELL_LENGTH / 2) / BETA_REF
    phase_center = -jnp.pi / 2 + RF_PHASE_OFFSET - omega * ct_center / u.c_light

    def cavity(index: int) -> EMTensorField:
        return TransformEMField(
            transform=Transform.make_translation(z=CELL_LENGTH / 2 + index * RF_PITCH),
            field=PillboxCavity(
                length=RF_LENGTH,
                frequency=RF_FREQUENCY,
                E0=RF_E0,
                mode="TM",
                m=0,
                n=1,
                p=0,
                phase=phase_center + index * jnp.pi,
            ),
        )

    return SumField(
        [
            coil(COIL_Z, COIL_JPHI),
            coil(CELL_LENGTH - COIL_Z, -COIL_JPHI),
            cavity(-1),
            cavity(0),
            cavity(1),
        ]
    )


def setup_scene(num_cells: int, num_pad: int = 2) -> EMTensorField:
    """Tile identical cells along z

    Cell 0 starts at z=0. An additional num_pad cells are placed upstream and
    downstream so that the fringe fields of the tracked region are correct.
    Each cell is also shifted in time by the reference particle arrival time, so
    every cell has the same RF phase relative to the reference particle.
    """

    def place(index: jax.Array) -> EMTensorField:
        z0 = index * CELL_LENGTH
        return TransformEMField(
            transform=Transform.make_translation(z=z0, ct=z0 / BETA_REF),
            field=setup_cell(),
        )

    indices = jnp.arange(-num_pad, num_cells + num_pad)
    return StackedField(fields=eqx.filter_vmap(place)(indices))


def sample_particles(rng: jax.Array, num_particles: int) -> MuonStateDz:
    """Sample an uncorrelated Gaussian beam at z=0

    With alpha=0 and zero kinetic angular momentum, the 4D transverse covariance
    in (x, px_kin, y, py_kin) is diagonal, with
        <x^2> = eps m beta / p,   <px^2> = eps m p / beta
    where eps is the normalized transverse emittance.
    """
    mass = REF_MUON.mass
    e_ref = REF_MUON.kin.t.ct
    sigma_x = jnp.sqrt(EMITTANCE_T * mass * BETA_T / P_REF)
    sigma_px = jnp.sqrt(EMITTANCE_T * mass * P_REF / BETA_T)
    sigmas = jnp.array(
        [sigma_x, sigma_x, sigma_px, sigma_px, SIGMA_T * u.c_light, SIGMA_E]
    )

    def sample_one(key: jax.Array) -> MuonStateDz:
        x, y, px, py, ct, dE = jax.random.normal(key, (6,)) * sigmas
        energy = e_ref + dE
        pz = jnp.sqrt(energy**2 - mass**2 - px**2 - py**2)
        return MuonStateDz.make(
            position=Cartesian4.make(x=x, y=y, ct=ct),
            momentum=Cartesian3.make(x=px, y=py, z=pz),
            q=1,
        )

    return jax.vmap(sample_one)(jax.random.split(rng, num_particles))


def render_scene_usd(scene: EMTensorField, path: Path) -> None:
    """Render the scene to a USDZ file

    Trajectories are not drawn: the beam is too small to see at the scale of the coils
    """
    from beamline.jax.export.usd import add_volume, make_stage, save_usdz

    stage = make_stage(str(path))
    add_volume(stage, "/beamline", scene)
    save_usdz(stage)


def plot_panels(
    series: dict[str, np.ndarray],
    columns: list[list[tuple[str, str]]],
    zvals: np.ndarray,
    title: str,
    path: Path,
):
    """Grid of per-z quantities, one panel (and y axis) each, sharing the z axis

    Args:
        series: Values at each z, by key
        columns: For each column of the grid, the (key, y label) of each row
    """
    ink, muted, color = "#0b0b0b", "#52514e", "#2a78d6"
    nrows = max(len(col) for col in columns)
    fig, axes = plt.subplots(
        nrows,
        len(columns),
        figsize=(6 * len(columns), 2.4 * nrows),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    fig.set_facecolor("#fcfcfb")
    for icol, col in enumerate(columns):
        for irow, ax in enumerate(axes[:, icol]):
            if irow >= len(col):
                ax.set_visible(False)
                continue
            key, label = col[irow]
            ax.set_facecolor("#fcfcfb")
            for zb in np.arange(0.0, zvals[-1] + 1.0, CELL_LENGTH):
                ax.axvline(zb / u.m, color=muted, lw=0.5, alpha=0.3)
            ax.plot(zvals / u.m, series[key], color=color, lw=2)
            ax.set_ylabel(label, color=ink, fontsize=13)
            ax.grid(axis="y", color=muted, lw=0.5, alpha=0.2)
            ax.tick_params(colors=muted, labelcolor=ink, labelsize=11)
            ax.ticklabel_format(axis="y", useOffset=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(muted)
        axes[len(col) - 1, icol].set_xlabel("z [m]", color=ink, fontsize=13)
    fig.suptitle(
        f"{title}  (vertical lines: cell boundaries)",
        color=ink,
        fontsize=14,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_benchmark_4p1_cell(artifacts_dir):
    num_cells = 5
    scene = setup_scene(num_cells=num_cells)
    num_particles = 1000
    initial_particles = sample_particles(
        rng=jax.random.PRNGKey(0), num_particles=num_particles
    )
    zvals = jnp.linspace(0.0, num_cells * CELL_LENGTH, num_cells * 40 + 1)

    aperture = BeamPipe(radius=BEAM_PIPE_RADIUS)

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, None))
    def track(field, start, cts):
        return diffrax_solve(field, start, cts, aperture=aperture)

    trajectories, stats = track(scene, initial_particles, zvals)
    assert trajectories.kin.p.z.shape == (num_particles, len(zvals))
    survived = trajectories.is_alive()[:, -1]
    assert jnp.all(stats["lost"] == ~survived)
    assert jnp.allclose(trajectories.kin.p.z[survived, -1], zvals[-1])
    # lost particles are frozen at their exit point, just outside the pipe
    exit_radius = jnp.hypot(trajectories.kin.p.x, trajectories.kin.p.y)[~survived, -1]
    assert jnp.all(exit_radius > BEAM_PIPE_RADIUS)

    stats = {"transmission": survived.mean(), **stats}
    with open(artifacts_dir / "benchmark_4p1_stats.json", "w") as fout:
        json.dump({k: np.asarray(v).tolist() for k, v in stats.items()}, fout)
    write_states_csv(
        artifacts_dir / "benchmark_4p1_trajectories.csv",
        trajectories,
        index_names=("particle", "step"),
    )

    # moments of the surviving beam at each z (states batched as particle, z)
    moments = jax.vmap(beam_moments, in_axes=(1, None))(trajectories, survived)
    moments = {k: np.asarray(v) for k, v in moments.items()}
    with open(artifacts_dir / "benchmark_4p1_moments.csv", "w") as fout:
        fout.write(",".join(["z", *moments]) + "\n")
        for i, z in enumerate(np.asarray(zvals)):
            fout.write(
                ",".join(f"{v:.6g}" for v in [z, *(m[i] for m in moments.values())])
            )
            fout.write("\n")

    zs = np.asarray(zvals)
    # RF phase of the mean arrival time relative to the design timing (constant
    # reference velocity) used to phase the cells
    phase_slip = 360.0 * RF_FREQUENCY * (moments["ct_mean"] - zs / BETA_REF) / u.c_light
    plot_panels(
        {
            "beta_t": moments["beta_t"] / u.mm,
            "alpha_t": moments["alpha_t"],
            "l_kin": moments["l_kin"] / (u.mm * u.MeV),
            "eps_t": moments["eps_t"] / u.micrometer,
            "survivors": np.asarray(trajectories.is_alive()).mean(axis=0),
            "pz_mean": moments["pz_mean"] / u.MeV,
            "energy_rms": moments["energy_rms"] / u.MeV,
            "ct_rms": moments["ct_rms"] / u.c_light / u.ps,
            "eps_l": moments["eps_l"] / u.micrometer,
            "phase_slip": phase_slip,
        },
        [
            [
                ("beta_t", "β⊥ [mm]"),
                ("alpha_t", r"$\alpha_\perp$"),
                ("l_kin", "L_kin [mm MeV/c]"),
                ("eps_t", "ε⊥ [µm]"),
                ("survivors", "Survivor fraction"),
            ],
            [
                ("pz_mean", "⟨pz⟩ [MeV/c]"),
                ("energy_rms", r"$\sigma_E$ [MeV]"),
                ("ct_rms", r"$\sigma_t$ [ps]"),
                ("eps_l", "εL [µm]"),
                ("phase_slip", "RF phase slip [deg]"),
            ],
        ],
        zs,
        "Benchmark 4.1: low emittance, no absorber",
        artifacts_dir / "benchmark_4p1_beam.png",
    )

    # linear optics with no absorber: transverse emittance is conserved
    assert moments["eps_t"][-1] == pytest.approx(moments["eps_t"][0], rel=5e-3)
    # matched beam at the start (sampling noise on 1000 particles is ~3%)
    assert moments["beta_t"][0] == pytest.approx(BETA_T, rel=0.15)
    assert moments["alpha_t"][0] == pytest.approx(0.0, abs=0.15)

    if importlib.util.find_spec("pxr") is not None:
        render_scene_usd(scene, artifacts_dir / "benchmark_4p1.usdz")
