"""Benchmark 3.1: muon distributions in kinetic energy and x' after an absorber.

Table 1 of the cooling code benchmarking note.
"""

import cooling_common as cc
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrate.stochastic import stochastic_solve
from beamline.jax.kinematics import MuonStateDz

# (material key, thickness, momentum, max stopped fraction).
ABSORBERS = [
    ("lithium_hydride_MICE", 65.37 * u.mm, 171.55 * u.MeV, 0.02),
    ("lithium_hydride_MICE", 65.37 * u.mm, 199.93 * u.MeV, 0.02),
    ("lithium_hydride_MICE", 65.37 * u.mm, 239.76 * u.MeV, 0.02),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 164.9 * u.MeV, 0.02),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 199.0 * u.MeV, 0.02),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 237.1 * u.MeV, 0.02),
    ("liquid_hydrogen_H2", 10.0 * u.mm, 30.0 * u.MeV, 1.0),
]

N_PARTICLES = 1_000_000
CHUNK = 50_000  # divides N_PARTICLES, so every chunk compiles to one shape
SEED = 20240701
N_BINS = 400
N_PROFILE = 100_000
N_S = 21  # depth points through the absorber
STEP_COUNTS = [1, 10, 100]
N_STEPS_PARTICLES = 1_000_000
THETA_EDGES = np.linspace(-200.0e-3, 200.0e-3, 33)


@pytest.mark.extended
@pytest.mark.parametrize(
    ("material_key", "thickness", "momentum", "max_stopped"),
    ABSORBERS,
    ids=[f"{k}_{p / u.MeV:g}MeV" for k, _, p, _ in ABSORBERS],
)
def test_benchmark_3p1_absorber(
    artifacts_dir, material_key, thickness, momentum, max_stopped
):
    """Particles start on axis along +z and are histogrammed downstream.

    Single-interval save grid [-thickness, +thickness]: an interior save point
    would force a sub-interval boundary inside the absorber.
    """
    absorber = AbsorberCylinder(
        material=MATERIALS[material_key],
        radius=250.0 * u.mm,
        length=thickness,
    )
    start = MuonStateDz.make(
        position=Cartesian4.make(z=-thickness),
        momentum=Cartesian3.make(z=momentum),
        q=1,
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    zs = jnp.array([-thickness, thickness])

    # Analytic predictions from the same material entry.
    params = absorber.interaction_params(start, thickness)
    mean_dE = float(params.mean_energy_loss)
    theta0 = float(params.theta0)
    T_in = float(start.kin.t.ct) - float(start.mass)

    @jax.jit
    @jax.vmap
    def run(key):
        # Return three scalars per particle rather than the saved state
        # to conserve memory.
        ys, _ = stochastic_solve(field, absorber, start, zs, key, kick=cc.KICK)
        end = jax.tree.map(lambda a: a[-1], ys)
        return end.kin.t.ct - start.mass, end.kin.t.x / end.kin.t.z, end.kin.p.z

    # Clamp at zero: 3*mean_dE exceeds the kinetic energy when the absorber is
    # comparable to the muon range.
    T_edges = np.linspace(max(0.0, T_in - 3.0 * mean_dE), T_in, N_BINS + 1)
    xp_edges = np.linspace(-6.0 * theta0, 6.0 * theta0, N_BINS + 1)
    T_counts = np.zeros(N_BINS, dtype=np.int64)
    xp_counts = np.zeros(N_BINS, dtype=np.int64)
    n_outside = 0
    n_stopped = 0

    keys = jr.split(jr.key(SEED), N_PARTICLES)
    for i in range(0, N_PARTICLES, CHUNK):
        T_out, xp, z_end = run(keys[i : i + CHUNK])
        z_end, T_out, xp = np.asarray(z_end), np.asarray(T_out), np.asarray(xp)
        # Particles floored at the rest mass by apply_energy_loss have pz -> 0; 
        # Count, exclude, and report them.
        arrived = (
            np.isfinite(z_end)
            & np.isfinite(T_out)
            & np.isfinite(xp)
            & (np.abs(z_end - thickness) < 1e-3 * thickness)
        )
        n_stopped += int((~arrived).sum())
        T_out, xp = T_out[arrived], xp[arrived]
        T_counts += np.histogram(T_out, bins=T_edges)[0]
        xp_counts += np.histogram(xp, bins=xp_edges)[0]
        n_outside += int(
            np.sum((T_out < T_edges[0]) | (T_out > T_edges[-1]))
            + np.sum((xp < xp_edges[0]) | (xp > xp_edges[-1]))
        )

    stopped_frac = n_stopped / N_PARTICLES
    assert stopped_frac < max_stopped, (
        f"{stopped_frac:.3%} stopped, above the {max_stopped:.1%} bound for "
        "this configuration"
    )

    stem = f"benchmark_3p1_{material_key}_{momentum / u.MeV:.2f}MeV"
    mat = absorber.material
    with open(artifacts_dir / f"{stem}.csv", "w") as f:
        f.write(f"# material={mat.name}\n")
        f.write(f"# density_g_cm3={mat.density / (u.g / u.cm3):.5f}\n")
        f.write(f"# radiation_length_g_cm2={mat.radiation_length / (u.g / u.cm2):.3f}\n")
        f.write(f"# mean_excitation_eV={mat.mean_excitation / u.eV:.2f}\n")
        if material_key == "lithium_hydride_MICE":
            f.write(
                "# NOTE density is the PDG tabulated 0.82 g/cm3; benchmark PDF "
                "Table 1 specifies 0.69 for LiH (+17.9% dE, +9.8% theta0)\n"
            )
        f.write(f"# thickness_mm={thickness / u.mm:.3f}\n")
        f.write(f"# momentum_MeV={momentum / u.MeV:.3f}\n")
        f.write(f"# n_particles={N_PARTICLES}\n")
        f.write(f"# char_length_mm={absorber.char_length / u.mm:.3f}\n")
        f.write(f"# predicted_mean_dE_MeV={mean_dE:.5f}\n")
        f.write(f"# predicted_theta0_mrad={theta0 * 1e3:.5f}\n")
        f.write(f"# straggling_sampler={cc.STRAGGLING_SAMPLER.__name__}\n")
        f.write(f"# scattering_sampler={cc.SCATTERING_SAMPLER.__name__}\n")
        f.write(f"# transmission={1.0 - stopped_frac:.6f}\n")
        f.write(f"# n_outside_histogram_range={n_outside}\n")
        f.write(f"# n_stopped_in_absorber={n_stopped}\n")
        f.write(f"# stopped_fraction={stopped_frac:.6f}\n")
        f.write("T_lo_MeV,T_hi_MeV,T_count,xprime_lo,xprime_hi,xprime_count\n")
        for j in range(N_BINS):
            f.write(
                f"{T_edges[j]:.6f},{T_edges[j + 1]:.6f},{T_counts[j]},"
                f"{xp_edges[j]:.9f},{xp_edges[j + 1]:.9f},{xp_counts[j]}\n"
            )

    # Plot results  
    T_mid = 0.5 * (T_edges[:-1] + T_edges[1:])
    xp_mid = 0.5 * (xp_edges[:-1] + xp_edges[1:])

    fig, (axT, axX) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Benchmark 3.1: {momentum / u.MeV:.2f} MeV/c muons through "
        f"{thickness / u.mm:.2f} mm {mat.name}"
    )

    axT.step(T_mid, T_counts, where="mid", color="#4c72b0")
    axT.axvline(
        T_in - mean_dE,
        color="#c44e52",
        lw=2,
        label=f"predicted mean loss = {mean_dE:.3f} MeV",
    )
    axT.set(xlabel="kinetic energy [MeV]", ylabel="count")
    axT.legend(fontsize=8)
    axT.grid(alpha=0.3)
    axX.step(xp_mid * 1e3, xp_counts, where="mid", color="#937860")
    axX.set(xlabel="x' [mrad]", ylabel="count")
    # Near the range limit almost nothing arrives, so skip the reference curve
    # and the log axis rather than dividing by a zero maximum.
    if xp_counts.max() > 0:
        gauss = np.exp(-0.5 * (xp_mid / theta0) ** 2)
        axX.plot(
            xp_mid * 1e3,
            gauss * xp_counts.max() / gauss.max(),
            color="k",
            lw=2,
            label=f"Highland $\\theta_0$ = {theta0 * 1e3:.3f} mrad",
        )
        axX.set_yscale("log")
        axX.legend(fontsize=8)
    axX.grid(alpha=0.3)
    axX.annotate(
        f"stopped in absorber: {stopped_frac:.3%}\n"
        f"transmission: {1.0 - stopped_frac:.3%}",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(artifacts_dir / f"{stem}.png", dpi=130)
    plt.close(fig)

SLIDE_PANELS = [
    ("s8",  "lithium_hydride_MICE", 65.37 * u.mm,  95.8, 0.02),
    ("s8",  "lithium_hydride_MICE", 65.37 * u.mm, 120.5, 0.02),
    ("s8",  "lithium_hydride_MICE", 65.37 * u.mm, 156.4, 0.02),
    ("s12", "lithium_hydride_MICE", 65.37 * u.mm,  96.0, 0.02),
    ("s12", "lithium_hydride_MICE", 65.37 * u.mm, 156.0, 0.02),
    ("s6",  "liquid_hydrogen_H2", 349.6 * u.mm,  90.2, 0.02),
    ("s6",  "liquid_hydrogen_H2", 349.6 * u.mm, 120.4, 0.02),
    ("s6",  "liquid_hydrogen_H2", 349.6 * u.mm, 153.9, 0.02),
    ("s12", "liquid_hydrogen_H2", 349.6 * u.mm,  90.0, 0.02),
    ("s12", "liquid_hydrogen_H2", 349.6 * u.mm, 120.0, 0.02),
    ("s12", "liquid_hydrogen_H2", 349.6 * u.mm, 156.0, 0.02),
    ("s6",  "liquid_hydrogen_H2",  10.0 * u.mm,   4.2, 1.0),
    ("s12", "liquid_hydrogen_H2",  10.0 * u.mm,   4.0, 1.0),
]

@pytest.mark.extended
@pytest.mark.parametrize(
    ("slide", "material_key", "thickness", "T_in", "max_stopped"),
    SLIDE_PANELS,
    ids=[f"{s}_{k.split('_')[0]}_{t:g}MeV" for s, k, _, t, _ in SLIDE_PANELS],
)
def test_benchmark_3p1_profiles(
    artifacts_dir, slide, material_key, thickness, T_in, max_stopped
):
    """Mean kinetic energy and RMS x-scattering angle vs depth.

    Momentum is derived from the slide's E_Kin(0) so the panel annotations
    match exactly: p = sqrt((T + m)^2 - m^2).
    """
    absorber = AbsorberCylinder(
        material=MATERIALS[material_key], radius=250.0 * u.mm, length=thickness
    )
    mass = float(
        MuonStateDz.make(
            position=Cartesian4.make(),
            momentum=Cartesian3.make(z=200.0 * u.MeV),
            q=1,
        ).mass
    )
    momentum = float(np.sqrt((T_in * u.MeV + mass) ** 2 - mass**2))
    start = MuonStateDz.make(
        position=Cartesian4.make(z=-thickness),
        momentum=Cartesian3.make(z=momentum),
        q=1,
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())

    # The cylinder is centred on the origin, so depth s into the material maps
    # to z = s - thickness/2. Prepend the start plane for the vacuum drift in.
    s_grid = jnp.linspace(0.0, thickness, N_S)
    zs = jnp.concatenate([jnp.array([-thickness]), s_grid - thickness / 2])

    @jax.jit
    @jax.vmap
    def run(key):
        ys, _ = stochastic_solve(field, absorber, start, zs, key, kick=cc.KICK)
        T = ys.kin.t.ct - start.mass
        thx = ys.kin.t.x / ys.kin.t.z
        return T[1:], thx[1:], ys.kin.p.z[1:]

    n = np.zeros(N_S)
    sT = np.zeros(N_S)
    sT2 = np.zeros(N_S)
    sth2 = np.zeros(N_S)
    z_want = np.asarray(zs[1:])

    T_all, th_all, ok_all = [], [], []
    keys = jr.split(jr.key(SEED + 1 + int(T_in * 10)), N_PROFILE)
    for i in range(0, N_PROFILE, CHUNK):
        T, thx, z = (np.asarray(a) for a in run(keys[i : i + CHUNK]))
        ok = (
            np.isfinite(T) & np.isfinite(thx)
            & (np.abs(z - z_want) < 1e-3 * thickness)
        )
        T_all.append(T); th_all.append(thx); ok_all.append(ok)
    T = np.concatenate(T_all); thx = np.concatenate(th_all)
    ok = np.concatenate(ok_all)

    n = ok.sum(axis=0)
    mean_T = np.array([T[ok[:, j], j].mean() for j in range(N_S)])
    # The Landau mean is divergent. Compare on median and mode.
    med_T = np.array([np.median(T[ok[:, j], j]) for j in range(N_S)])
    mode_T = np.empty(N_S)
    for j in range(N_S):
        v = T[ok[:, j], j]
        h, e = np.histogram(v, bins=200)
        mode_T[j] = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    sigma_T = np.array([T[ok[:, j], j].std() for j in range(N_S)])
    rms_thx = np.array([np.sqrt((thx[ok[:, j], j] ** 2).mean()) for j in range(N_S)])

    stem = f"benchmark_3p1_profile_{slide}_{material_key}_{T_in:g}MeV"
    mat = absorber.material
    with open(artifacts_dir / f"{stem}.csv", "w") as f:
        f.write(f"# slide={slide}\n")
        f.write(f"# material={mat.name}\n")
        f.write(f"# density_g_cm3={mat.density / (u.g / u.cm3):.5f}\n")
        f.write(f"# radiation_length_g_cm2={mat.radiation_length / (u.g / u.cm2):.3f}\n")
        f.write(f"# thickness_mm={thickness / u.mm:.3f}\n")
        f.write(f"# momentum_MeV={momentum / u.MeV:.3f}\n")
        f.write(f"# T_in_MeV={float(start.kin.t.ct) - float(start.mass):.4f}\n")
        f.write(f"# n_particles={N_PROFILE}\n")
        f.write(f"# straggling_sampler={cc.STRAGGLING_SAMPLER.__name__}\n")
        f.write(f"# scattering_sampler={cc.SCATTERING_SAMPLER.__name__}\n")
        f.write("s_mm,n_alive,mean_T_MeV,median_T_MeV,mode_T_MeV,sigma_T_MeV,rms_thetax_mrad\n")
        for j in range(N_S):
            f.write(
                f"{float(s_grid[j]) / u.mm:.4f},{int(n[j])},"
                f"{mean_T[j]:.6f},{med_T[j]:.6f},{mode_T[j]:.6f},"
                f"{sigma_T[j]:.6f},{rms_thx[j] * 1e3:.6f}\n"
            )

@pytest.mark.extended
@pytest.mark.parametrize("n_steps", STEP_COUNTS)
def test_benchmark_3p1_step_scan(artifacts_dir, n_steps):
    """theta_x distribution vs number of in-material integration steps.

    char_length = thickness / n_steps reproduces their "steps" axis. A
    convolutional scattering model gives the same distribution at every step
    count; one whose log term is re-evaluated per segment does not.
    """
    thickness, momentum = 349.6 * u.mm, 164.9 * u.MeV
    absorber = AbsorberCylinder(
        material=MATERIALS["liquid_hydrogen_H2"],
        radius=250.0 * u.mm,
        length=thickness,
        char_length=thickness / n_steps,
    )
    start = MuonStateDz.make(
        position=Cartesian4.make(z=-thickness),
        momentum=Cartesian3.make(z=momentum),
        q=1,
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    zs = jnp.array([-thickness, thickness])

    @jax.jit
    @jax.vmap
    def run(key):
        ys, _ = stochastic_solve(field, absorber, start, zs, key, kick=cc.KICK)
        end = jax.tree.map(lambda a: a[-1], ys)
        return end.kin.t.x / end.kin.t.z, end.kin.p.z

    counts = np.zeros(len(THETA_EDGES) - 1, dtype=np.int64)
    n_stopped = 0
    keys = jr.split(jr.key(SEED + 2), N_STEPS_PARTICLES)
    for i in range(0, N_STEPS_PARTICLES, CHUNK):
        thx, z_end = (np.asarray(a) for a in run(keys[i : i + CHUNK]))
        ok = np.isfinite(thx) & (np.abs(z_end - thickness) < 1e-3 * thickness)
        n_stopped += int((~ok).sum())
        counts += np.histogram(thx[ok], bins=THETA_EDGES)[0]

    params = absorber.interaction_params(start, thickness)
    with open(artifacts_dir / f"benchmark_3p1_steps_{n_steps:04d}.csv", "w") as f:
        f.write(f"# n_steps={n_steps}\n")
        f.write(f"# char_length_mm={thickness / n_steps / u.mm:.4f}\n")
        f.write(f"# n_particles={N_STEPS_PARTICLES}\n")
        f.write(f"# n_stopped={n_stopped}\n")
        f.write(f"# predicted_theta0_mrad={float(params.theta0) * 1e3:.5f}\n")
        f.write(f"# scattering_sampler={cc.SCATTERING_SAMPLER.__name__}\n")
        f.write("theta_lo_mrad,theta_hi_mrad,count\n")
        for j in range(len(counts)):
            f.write(
                f"{THETA_EDGES[j] * 1e3:.4f},{THETA_EDGES[j + 1] * 1e3:.4f},"
                f"{counts[j]}\n"
            )