"""Cooling code benchmark tests.

As described in:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf
"""

from collections.abc import Callable

import diffrax
import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.scattering import highland_scattering_sampler
from beamline.jax.absorber.straggling import landau_energy_loss_sampler
from beamline.jax.absorber.volume import AbsorberCylinder, SumMaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import (
    EMTensorField,
    SimpleEMField,
    SumField,
    TransformEMField,
)
from beamline.jax.integrate.propagate import diffrax_solve, particle_interaction
from beamline.jax.integrate.stochastic import (
    StochasticKick,
    energy_loss_kick,
    scattering_kick,
    stochastic_solve,
)
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity
from beamline.jax.types import SFloat

SOLENOID = ThickSolenoid(
    Rin=250.0 * u.mm,
    Rout=419.3 * u.mm,
    jphi=500.0 * u.A / u.mm**2,
    L=140.0 * u.mm,
)

ABSORBERS = [
    ("lithium_hydride_LiH", 65.37 * u.mm, 171.55 * u.MeV),
    ("lithium_hydride_LiH", 65.37 * u.mm, 199.93 * u.MeV),
    ("lithium_hydride_LiH", 65.37 * u.mm, 239.76 * u.MeV),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 164.9 * u.MeV),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 199.0 * u.MeV),
    ("liquid_hydrogen_H2", 349.6 * u.mm, 237.1 * u.MeV)
]

N_PARTICLES_3P1 = 1_000_000
CHUNK_3P1 = 50_000
SEED_3P1 = 20240701
N_BINS_3P1 = 400

KICK_3P1 = StochasticKick(
    straggling=energy_loss_kick(landau_energy_loss_sampler),
    scattering=scattering_kick(highland_scattering_sampler),
)

@pytest.mark.extended
@pytest.mark.parametrize(
    ("material_key", "thickness", "momentum"),
    ABSORBERS,
    ids=[f"{k}_{p / u.MeV:g}MeV" for k, _, p in ABSORBERS],
)
def test_benchmark_3p1_absorber(artifacts_dir, material_key, thickness, momentum):
    """Benchmark 3.1: distributions in kinetic energy and x' through an absorber.

    Parameters from Table 1. Particles start on axis travelling along +z and
    are histogrammed at a plane downstream of the absorber.

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

    params = absorber.interaction_params(start, thickness)
    mean_dE = float(params.mean_energy_loss)
    theta0 = float(params.theta0)
    T_in = float(start.kin.t.ct) - float(start.mass)

    @jax.jit
    @jax.vmap
    def run(key):
        ys, _ = stochastic_solve(field, absorber, start, zs, key, kick=KICK_3P1)
        end = jax.tree.map(lambda a: a[-1], ys)
        T_out = end.kin.t.ct - start.mass
        xp = end.kin.t.x / end.kin.t.z
        return T_out, xp, end.kin.p.z

    # Fixed edges so per-chunk counts are summable.
    T_edges = np.linspace(T_in - 3.0 * mean_dE, T_in, N_BINS_3P1 + 1)
    xp_edges = np.linspace(-6.0 * theta0, 6.0 * theta0, N_BINS_3P1 + 1)
    T_counts = np.zeros(N_BINS_3P1, dtype=np.int64)
    xp_counts = np.zeros(N_BINS_3P1, dtype=np.int64)
    n_outside = 0
    n_stopped = 0

    keys = jr.split(jr.key(SEED_3P1), N_PARTICLES_3P1)
    for i in range(0, N_PARTICLES_3P1, CHUNK_3P1):
        T_out, xp, z_end = run(keys[i : i + CHUNK_3P1])
        z_end = np.asarray(z_end)
        T_out = np.asarray(T_out)
        xp = np.asarray(xp)
        """Particles floored at the rest mass by apply_energy_loss have pz -> 0;
        MuonStateDz integrates d/dz with scale = E/pz, so the RHS diverges and
        they never reach the end. These are muons stopped in the
        absorber by the unbounded Landau tail. This counts, excludes them and
        report the count.
        """
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

    stopped_frac = n_stopped / N_PARTICLES_3P1
    assert stopped_frac < 0.02, (
        f"{stopped_frac:.3%} of particles stopped in the absorber; "
        "expected < 1% from landau tail"
    )

    stem = f"benchmark_3p1_{material_key}_{momentum / u.MeV:.2f}MeV"
    mat = absorber.material
    with open(artifacts_dir / f"{stem}.csv", "w") as f:
        f.write(f"# material={mat.name}\n")
        f.write(f"# density_g_cm3={mat.density / (u.g / u.cm3):.5f}\n")
        f.write(f"# radiation_length_g_cm2={mat.radiation_length / (u.g / u.cm2):.3f}\n")
        f.write(f"# mean_excitation_eV={mat.mean_excitation / u.eV:.2f}\n")
        if material_key == "lithium_hydride_LiH":
            f.write(
                "# NOTE density is the PDG tabulated 0.82 g/cm3; benchmark PDF "
                "Table 1 specifies 0.69 for LiH (+17.9% dE, +9.8% theta0)\n"
            )
        f.write(f"# thickness_mm={thickness / u.mm:.3f}\n")
        f.write(f"# momentum_MeV={momentum / u.MeV:.3f}\n")
        f.write(f"# n_particles={N_PARTICLES_3P1}\n")
        f.write(f"# char_length_mm={absorber.char_length / u.mm:.3f}\n")
        f.write(f"# predicted_mean_dE_MeV={mean_dE:.5f}\n")
        f.write(f"# predicted_theta0_mrad={theta0 * 1e3:.5f}\n")
        f.write(f"# n_outside_histogram_range={n_outside}\n")
        f.write(f"# n_stopped_in_absorber={n_stopped}\n")
        f.write(f"# stopped_fraction={stopped_frac:.6f}\n")
        f.write("T_lo_MeV,T_hi_MeV,T_count,xprime_lo,xprime_hi,xprime_count\n")
        for j in range(N_BINS_3P1):
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
    gauss = np.exp(-0.5 * (xp_mid / theta0) ** 2)
    axX.plot(
        xp_mid * 1e3,
        gauss * xp_counts.max() / gauss.max(),
        color="k",
        lw=2,
        label=f"Highland $\\theta_0$ = {theta0 * 1e3:.3f} mrad",
    )
    axX.set(xlabel="x' [mrad]", ylabel="count")
    axX.set_yscale("log")
    axX.legend(fontsize=8)
    axX.grid(alpha=0.3)
    axX.annotate(
        f"stopped in absorber: {stopped_frac:.3%}\n(unbounded Landau tail)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(artifacts_dir / f"{stem}.png", dpi=130)
    plt.close(fig)


def test_benchmark_3p2_solenoid(artifacts_dir):
    """Benchmark 3.2: Muon through solenoid

    Parameters from Table 2
    """
    xpos = jnp.arange(-200.0 * u.mm, 201.0 * u.mm, 10.0 * u.mm)
    xpos = xpos.at[xpos == 0.0].set(1e-6)
    zs = jnp.linspace(-500.0 * u.mm, 500.0 * u.mm, 100)

    @jax.jit
    def run(fieldobj: ThickSolenoid, xstart: SFloat) -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(x=xstart, z=-500.0 * u.mm),
            momentum=Cartesian3.make(z=200 * u.MeV),
            q=1,
        )
        sol, _ = diffrax_solve(fieldobj, start, zs, forward_mode=True)
        return sol

    track: MuonStateDz = jax.vmap(run, in_axes=(None, 0))(SOLENOID, xpos)
    end: MuonStateDz = jax.tree.map(lambda x: x[:, -1], track)
    # TODO: understand why forward_mode=True fails here (when using jacfwd) at x=0.0
    grad: MuonStateDz = jax.vmap(jax.jacfwd(run), in_axes=(None, 0))(SOLENOID, xpos)
    assert track.kin.p.x.shape == (len(xpos), len(zs))

    def extract(
        dstate: MuonStateDz, get: Callable[[ThickSolenoid], SFloat]
    ) -> MuonStateDz:
        return jax.tree.map(
            get,
            dstate,
            is_leaf=lambda x: isinstance(x, ThickSolenoid),
        )

    grad_Rin = extract(grad, lambda f: f.Rin)
    grad_Rout = extract(grad, lambda f: f.Rout)
    grad_jphi = extract(grad, lambda f: f.jphi)
    grad_L = extract(grad, lambda f: f.L)

    with open(artifacts_dir / "benchmark_3p2_solenoid_data.csv", "w") as f:
        f.write("xf,yf,zf,tf,pxf,pyf,pzf,Ef\n")
        cols = [
            end.kin.p.x / u.mm,
            end.kin.p.y / u.mm,
            end.kin.p.z / u.mm,
            end.kin.p.ct / u.mm,
            end.kin.t.x / u.MeV,
            end.kin.t.y / u.MeV,
            end.kin.t.z / u.MeV,
            end.kin.t.ct / u.MeV,
        ]
        for row in zip(*cols, strict=True):
            f.write(",".join(f"{val:.6f}" for val in row) + "\n")

    """Full trajectories (PDF section 2: "trajectories as a function of radial
    offset").
    """
    with open(artifacts_dir / "benchmark_3p2_solenoid_tracks.csv", "w") as f:
        f.write("particle,x0_mm,z_mm,x_mm,y_mm,ct_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n")
        for ip in range(len(xpos)):
            for iz in range(len(zs)):
                f.write(
                    f"{ip},{xpos[ip] / u.mm:.4f},"
                    f"{track.kin.p.z[ip, iz] / u.mm:.6f},"
                    f"{track.kin.p.x[ip, iz] / u.mm:.6f},"
                    f"{track.kin.p.y[ip, iz] / u.mm:.6f},"
                    f"{track.kin.p.ct[ip, iz] / u.mm:.6f},"
                    f"{track.kin.t.x[ip, iz] / u.MeV:.6f},"
                    f"{track.kin.t.y[ip, iz] / u.MeV:.6f},"
                    f"{track.kin.t.z[ip, iz] / u.MeV:.6f},"
                    f"{track.kin.t.ct[ip, iz] / u.MeV:.6f}\n"
                )

    fig, (axx, axy) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ip in range(0, len(xpos), 2):
        axx.plot(track.kin.p.z[ip] / u.mm, track.kin.p.x[ip] / u.mm, lw=0.8)
        axy.plot(track.kin.p.z[ip] / u.mm, track.kin.p.y[ip] / u.mm, lw=0.8)
    axx.set_ylabel("x [mm]")
    axy.set_ylabel("y [mm]")
    axy.set_xlabel("z [mm]")
    axx.set_title("Benchmark 3.2: trajectories vs radial offset")
    for a in (axx, axy):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "benchmark_3p2_solenoid_tracks.png", dpi=150)
    plt.close(fig)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 8))
    (dots,) = ax.plot(
        end.kin.p.x,
        end.kin.p.y,
        marker=".",
        color="k",
        ls="none",
        label="beamline",
    )

    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    ax.plot(
        SOLENOID.Rin * jnp.cos(theta),
        SOLENOID.Rin * jnp.sin(theta),
        ls="--",
        color="gray",
    )
    ax.plot(
        SOLENOID.Rout * jnp.cos(theta),
        SOLENOID.Rout * jnp.sin(theta),
        ls="--",
        color="gray",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title = ax.set_title("Benchmark 3.2: Muon through solenoid")
    ax.set_xlim(-SOLENOID.Rin, SOLENOID.Rin)
    ax.set_ylim(-SOLENOID.Rin, SOLENOID.Rin)
    ax.legend(loc="upper center")

    # ax.set_facecolor("none")
    # fig.set_facecolor("none")
    fig.savefig(artifacts_dir / "benchmark_3p2_solenoid.png", dpi=150)

    def get_framedata(frame: int):
        return {
            "x": track.kin.p.x[:, frame],
            "y": track.kin.p.y[:, frame],
            "grad_Rin_x": grad_Rin.kin.p.x[:, frame],
            "grad_Rin_y": grad_Rin.kin.p.y[:, frame],
            "grad_Rout_x": grad_Rout.kin.p.x[:, frame],
            "grad_Rout_y": grad_Rout.kin.p.y[:, frame],
            "grad_jphi_x": grad_jphi.kin.p.x[:, frame] * (u.A / u.mm**2),
            "grad_jphi_y": grad_jphi.kin.p.y[:, frame] * (u.A / u.mm**2),
            "grad_L_x": grad_L.kin.p.x[:, frame],
            "grad_L_y": grad_L.kin.p.y[:, frame],
        }

    scale = 0.5
    frame = 0
    framedata = get_framedata(frame)

    dots.set_data(framedata["x"], framedata["y"])
    q_Rin = ax.quiver(
        framedata["x"],
        framedata["y"],
        framedata["grad_Rin_x"],
        framedata["grad_Rin_y"],
        color="blue",
        label="d/dRin",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    q_Rout = ax.quiver(
        framedata["x"],
        framedata["y"],
        framedata["grad_Rout_x"],
        framedata["grad_Rout_y"],
        color="orange",
        label="d/dRout",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    q_jphi = ax.quiver(
        framedata["x"],
        framedata["y"],
        framedata["grad_jphi_x"],
        framedata["grad_jphi_y"],
        color="green",
        label="d/djphi",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    q_L = ax.quiver(
        framedata["x"],
        framedata["y"],
        framedata["grad_L_x"],
        framedata["grad_L_y"],
        color="red",
        label="d/dL",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    ax.legend(loc="upper center")

    def update_quiver(frame: int):
        framedata = get_framedata(frame)
        dots.set_data(framedata["x"], framedata["y"])
        offsets = jnp.stack((framedata["x"], framedata["y"]), axis=-1)
        q_Rin.set_offsets(offsets)
        q_Rin.set_UVC(framedata["grad_Rin_x"], framedata["grad_Rin_y"])
        q_Rout.set_offsets(offsets)
        q_Rout.set_UVC(framedata["grad_Rout_x"], framedata["grad_Rout_y"])
        q_jphi.set_offsets(offsets)
        q_jphi.set_UVC(framedata["grad_jphi_x"], framedata["grad_jphi_y"])
        q_L.set_offsets(offsets)
        q_L.set_UVC(framedata["grad_L_x"], framedata["grad_L_y"])
        title.set_text(f"3.2: Muon through solenoid (z={zs[frame]: 3.0f} mm)")
        return (dots, q_Rin, q_Rout, q_jphi, q_L, title)

    anim = FuncAnimation(
        fig,
        update_quiver,
        frames=len(zs),
        blit=True,
    )
    anim.save(
        artifacts_dir / "benchmark_3p2_solenoid_grads.gif", writer="pillow", fps=10
    )
    fig.savefig(artifacts_dir / "benchmark_3p2_solenoid_grads.png", dpi=300)


@pytest.fixture
def reference_benchmark_3p2_solenoid():
    solver = diffrax.Dopri8()
    stepsize = diffrax.PIDController(rtol=1e-10, atol=1e-12)
    dt0 = None
    xpos = jnp.arange(-200.0 * u.mm, 201.0 * u.mm, 10.0 * u.mm)
    zs = jnp.linspace(-500.0 * u.mm, 500.0 * u.mm, 2)

    def run(fieldobj: ThickSolenoid, xstart: SFloat) -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(x=xstart, z=-500.0 * u.mm),
            momentum=Cartesian3.make(z=200 * u.MeV),
            q=1,
        )
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(particle_interaction),
            solver=solver,
            t0=zs[0],
            t1=zs[-1],
            dt0=dt0,
            y0=start,
            args=fieldobj,
            saveat=diffrax.SaveAt(ts=zs),
            stepsize_controller=stepsize,
        )
        return jax.tree.map(lambda x: x[-1], sol.ys)

    return jax.vmap(run, in_axes=(None, 0))(SOLENOID, xpos)


@pytest.mark.extended
@pytest.mark.parametrize(
    ("nsolver", "nstepsize"),
    [
        ("dopri5", "constant1cm"),
        ("tsit5", "constant1cm"),
        ("heun", "constant1cm"),
        ("dopri5", "constant10cm"),
        ("dopri5", "pid_rtol1em7"),
        ("dopri5", "pid_rtol1em5"),
        ("dopri8", "pid_rtol1em10"),
        ("dopri5", "pid_rtol1em3"),
        ("heun", "pid_rtol1em3"),
    ],
)
def test_benchmark_3p2_solenoid_perf(
    benchmark,
    reference_benchmark_3p2_solenoid: MuonStateDz,
    nsolver: str,
    nstepsize: str,
):
    """dt0 and dx in mm"""
    nsolvers = {
        "dopri5": diffrax.Dopri5(),
        "dopri8": diffrax.Dopri8(),
        "tsit5": diffrax.Tsit5(),
        "heun": diffrax.Heun(),
    }
    solver = nsolvers[nsolver]
    # initial step doesn't matter much for adaptive solvers, 1 mm is a reasonable default
    # (setting to None lets diffrax pick its own but only adds a tiny overhead)
    nstepsizes = {
        "pid_rtol1em3": (1 * u.mm, diffrax.PIDController(rtol=1e-3, atol=1e-6)),
        "pid_rtol1em5": (1 * u.mm, diffrax.PIDController(rtol=1e-5, atol=1e-7)),
        "pid_rtol1em7": (1 * u.mm, diffrax.PIDController(rtol=1e-7, atol=1e-9)),
        "pid_rtol1em10": (1 * u.mm, diffrax.PIDController(rtol=1e-10, atol=1e-12)),
        "constant1cm": (1 * u.cm, diffrax.ConstantStepSize()),
        "constant10cm": (10 * u.cm, diffrax.ConstantStepSize()),
    }
    dt0, stepsize = nstepsizes[nstepsize]

    # TODO: a separate benchmark to see the scaling with len(xpos)
    # first check: 40 to 400 points was 5x
    xpos = jnp.arange(-200.0 * u.mm, 201.0 * u.mm, 10.0 * u.mm)
    zs = jnp.linspace(-500.0 * u.mm, 500.0 * u.mm, 2)

    def run(fieldobj: ThickSolenoid, xstart: SFloat) -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(x=xstart, z=-500.0 * u.mm),
            momentum=Cartesian3.make(z=200 * u.MeV),
            q=1,
        )
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(particle_interaction),
            solver=solver,
            t0=zs[0],
            t1=zs[-1],
            dt0=dt0,
            y0=start,
            args=fieldobj,
            saveat=diffrax.SaveAt(ts=zs),
            stepsize_controller=stepsize,
        )
        return jax.tree.map(lambda x: x[-1], sol.ys)

    def runbench() -> MuonStateDz:
        runvec = jax.jit(jax.vmap(run, in_axes=(None, 0)))
        return jax.block_until_ready(runvec(SOLENOID, xpos))

    result = runbench()

    def reduce(leaf_func, a, b) -> float:
        out = jax.tree.reduce(
            jnp.maximum,
            jax.tree.map(
                leaf_func,
                a,
                b,
            ),
            initializer=0.0,
        )
        return float(out)

    def abs_diff(a, b):
        return jnp.max(abs(a - b))

    def rel_diff(a, b):
        num = abs(a - b)
        den = b
        return jnp.max(jnp.where(den != 0, num / den, 1.0))

    benchmark.extra_info = {
        "max_abs_diff_pos": reduce(
            abs_diff,
            result.kin.p,
            reference_benchmark_3p2_solenoid.kin.p,
        ),
        "max_abs_diff_mom": reduce(
            abs_diff,
            result.kin.t,
            reference_benchmark_3p2_solenoid.kin.t,
        ),
        "max_rel_diff_pos": reduce(
            rel_diff,
            result.kin.p,
            reference_benchmark_3p2_solenoid.kin.p,
        ),
        "max_rel_diff_mom": reduce(
            rel_diff,
            result.kin.t,
            reference_benchmark_3p2_solenoid.kin.t,
        ),
    }
    benchmark(runbench)


def test_benchmark_3p3_rf_cavity(artifacts_dir, benchmark):
    """Benchmark 3.3: Muon through RF cavity

    Parameters from Table 3
    """
    frequency = 704.0 * u.MHz
    # time to get to 0 position is 500mm / beta*c
    ref_muon = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    t0 = 500 * u.mm / (ref_muon.beta() * u.c_light)
    # advance by pi/2 so we are in bunching mode
    # i.e. refence particle momentum is unchanged
    phase = -2 * u.pi * ((t0 * frequency - 0.25) % 1.0)

    cavity = PillboxCavity(
        length=183.6 * u.mm,
        frequency=frequency,
        E0=30.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=0,
        phase=phase,
    )

    X, CT = jnp.meshgrid(
        # TODO: fix singularity at origin
        jnp.linspace(0.0 * u.mm, 200.0 * u.mm, 21).at[0].set(1e-8 * u.mm),
        jnp.linspace(0.0 * u.mm, 1.0 / 0.704 * u.ns * u.c_light, 11),
        indexing="ij",
    )
    x, ct = X.flatten(), CT.flatten()
    starts = MuonStateDz.make(
        position=Cartesian4.make(x=x, z=-500.0 * u.mm, ct=ct),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    saveat = jnp.array([-500.0 * u.mm, 500.0 * u.mm])

    @jax.jit
    def run_one(start: MuonStateDz) -> tuple[MuonStateDz, dict]:
        ys, stats = diffrax_solve(
            field=cavity,
            start=start,
            cts=saveat,
            rtol=1e-7,
            atol=1e-9,
        )
        return jax.tree.map(lambda x: x[-1], ys), stats

    # start1 = jax.tree.map(lambda x: x[0], starts)
    # end = run_one(start1)
    # return

    def run_all(starts: MuonStateDz) -> tuple[MuonStateDz, dict]:
        # vmap seems faster by about 20%, this might be more dramatic when there are many volumes
        # _, (ends, stats) = jax.lax.scan(lambda _, s: (None, run_one(s)), None, starts)
        ends, stats = jax.vmap(run_one)(starts)
        ends = jax.tree.map(lambda x: x.reshape((*X.shape, -1)), ends)
        return ends, stats

    ends, stats = run_all(starts)

    # Trajectories on a fine z grid
    ztrack = jnp.linspace(-500.0 * u.mm, 500.0 * u.mm, 100)

    @jax.jit
    @jax.vmap
    def run_track(start: MuonStateDz) -> MuonStateDz:
        ys, _ = diffrax_solve(
            field=cavity, start=start, cts=ztrack, rtol=1e-7, atol=1e-9
        )
        return ys

    tracks = run_track(starts)  # leaves are (nx * nt, nz)

    with open(artifacts_dir / "benchmark_3p3_rf_tracks.csv", "w") as f:
        f.write(
            "particle,x0_mm,ct0_ns,z_mm,x_mm,y_mm,ct_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n"
        )
        for ip in range(len(x)):
            for iz in range(len(ztrack)):
                f.write(
                    f"{ip},{x[ip] / u.mm:.4f},{ct[ip] / (u.c_light * u.ns):.6f},"
                    f"{tracks.kin.p.z[ip, iz] / u.mm:.6f},"
                    f"{tracks.kin.p.x[ip, iz] / u.mm:.6f},"
                    f"{tracks.kin.p.y[ip, iz] / u.mm:.6f},"
                    f"{tracks.kin.p.ct[ip, iz] / u.mm:.6f},"
                    f"{tracks.kin.t.x[ip, iz] / u.MeV:.6f},"
                    f"{tracks.kin.t.y[ip, iz] / u.MeV:.6f},"
                    f"{tracks.kin.t.z[ip, iz] / u.MeV:.6f},"
                    f"{tracks.kin.t.ct[ip, iz] / u.MeV:.6f}\n"
                )

    nx, nt = X.shape
    E0 = float(ref_muon.kin.t.ct)

    # energy vs z, one line per radial offset, at t = 0
    fig, ax = plt.subplots(figsize=(8, 5))
    for ix in range(nx):
        ip = ix * nt + 0
        ax.plot(
            tracks.kin.p.z[ip] / u.mm,
            (tracks.kin.t.ct[ip] - E0) / u.MeV,
            lw=0.9,
            label=f"x={X[ix, 0] / u.mm:.0f} mm" if ix % 5 == 0 else None,
        )
    ax.set(
        xlabel="z [mm]",
        ylabel="E - E0 [MeV]",
        title="Benchmark 3.3: energy gain vs radial offset (t=0)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "benchmark_3p3_rf_tracks_radial.png", dpi=150)
    plt.close(fig)

    # energy vs z, one line per time offset, on axis
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in range(nt):
        ip = 0 * nt + it
        ax.plot(
            tracks.kin.p.z[ip] / u.mm,
            (tracks.kin.t.ct[ip] - E0) / u.MeV,
            lw=0.9,
            label=f"t={CT[0, it] / (u.c_light * u.ns):.3f} ns" if it % 2 == 0 else None,
        )
    ax.set(
        xlabel="z [mm]",
        ylabel="E - E0 [MeV]",
        title="Benchmark 3.3: energy gain vs time offset (on axis)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "benchmark_3p3_rf_tracks_time.png", dpi=150)
    plt.close(fig)

    # histogram of number of steps taken
    fig, ax = plt.subplots()
    ax.hist(stats["num_steps"], bins=20, label="total steps")
    ax.hist(stats["num_rejected_steps"], bins=20, alpha=0.5, label="rejected steps")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(artifacts_dir / "benchmark_3p3_rf_steps.png")

    # Plot results
    fig, ax = plt.subplots()

    for ix in [0, 10, 16, 20]:
        xstart = X[ix, 0]
        ctvals = CT[ix, :]
        Evals = ends.kin.t.ct[ix, :] - ref_muon.kin.t.ct

        ax.plot(
            (ctvals - ctvals.min()) / (u.c_light * u.ns),
            Evals / u.MeV,
            marker="^",
            color="green",
            ls="none",
            label=f"x={xstart} mm",
        )
        ax.set_xlabel("t-t0 [ns]")
        ax.set_ylabel("Delta E [MeV]")
        ax.set_title("Benchmark 3.3: Muon through RF cell")
        ax.legend()

        # ax.set_facecolor("none")
        # fig.set_facecolor("none")
        fig.savefig(artifacts_dir / f"benchmark_3p3_rf_x{ix:02d}.png")
        ax.clear()

    def bench_func():
        f = jax.jit(run_all)
        return jax.block_until_ready(f(starts))

    bench_func()
    benchmark(bench_func)


@pytest.mark.parametrize("forward", [True, False])
def test_benchmark_3p3_rf_tune(artifacts_dir, forward: bool):
    """Derivative with respect to cavity phase and z position

    Both should be able to accomplish the same thing if gradients are working

    Test that forward and reverse mode autodiff work (TODO: and give the same result)
    """
    frequency = 704.0 * u.MHz
    # time to get to 0 position is 500mm / beta*c
    ref_muon = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    t0 = 500 * u.mm / (ref_muon.beta() * u.c_light)
    # advance by pi/2 so we are in bunching mode
    # i.e. refence particle momentum is unchanged
    phase = -2 * u.pi * ((t0 * frequency - 0.25) % 1.0)

    def build_cavity(phase_shift: SFloat, z_shift: SFloat) -> EMTensorField:
        cavity = PillboxCavity(
            length=183.6 * u.mm,
            frequency=frequency,
            E0=30.0 * u.MV / u.m,
            mode="TM",
            m=0,
            n=1,
            p=0,
            phase=phase + phase_shift,
        )
        return TransformEMField(
            transform=Transform.make_translation(z=z_shift),
            field=cavity,
        )

    def track_ref(field: EMTensorField) -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(x=1e-8 * u.mm, z=-500.0 * u.mm),
            momentum=Cartesian3.make(z=200.0 * u.MeV),
            q=1,
        )
        saveat = jnp.array([-500.0 * u.mm, 500.0 * u.mm])
        ys, _ = diffrax_solve(
            field=field,
            start=start,
            cts=saveat,
            rtol=1e-7,
            atol=1e-9,
            forward_mode=forward,
        )
        return jax.tree.map(lambda x: x[-1], ys)

    @eqx.filter_jit
    def objective(phase_shift: SFloat, z_shift: SFloat) -> SFloat:
        field = build_cavity(phase_shift, z_shift)
        end = track_ref(field)
        return end.kin.t.ct

    jacobian = jax.jacfwd if forward else jax.jacrev

    phase_vals = jnp.linspace(-1.5, 1.5, 9)
    E_vals = []
    dEdphase_vals = []
    for phase_shift in phase_vals:
        Efinal = objective(phase_shift, 0.0)
        dEdphase = jacobian(objective, argnums=0)(phase_shift, 0.0)
        E_vals.append(Efinal)
        dEdphase_vals.append(dEdphase)

    fig, ax = plt.subplots()

    ax.plot(
        phase_vals,
        E_vals,
        marker="o",
        color="green",
        lw=1,
    )
    for phase_shift, Efinal, dEdphase in zip(
        phase_vals, E_vals, dEdphase_vals, strict=True
    ):
        p = jnp.linspace(phase_shift - 0.05, phase_shift + 0.05, 3)
        ax.plot(p, Efinal + dEdphase * (p - phase_shift), color="black")

    ax.set_xlabel("Phase shift [rad]")
    ax.set_ylabel("Final energy [MeV]")
    fwdlabel = "fwd" if forward else "rev"
    fig.savefig(artifacts_dir / f"benchmark_3p3_rf_phase_tune_{fwdlabel}.png")

    z_vals = jnp.linspace(-150, 150, 9)
    E_vals = []
    dEdz_vals = []
    for z_shift in z_vals:
        Efinal = objective(0.0, z_shift)
        dEdz = jacobian(objective, argnums=1)(0.0, z_shift)
        E_vals.append(Efinal)
        dEdz_vals.append(dEdz)

    fig, ax = plt.subplots()

    ax.plot(
        z_vals,
        E_vals,
        marker="o",
        color="green",
        lw=1,
    )
    for z_shift, Efinal, dEdz in zip(z_vals, E_vals, dEdz_vals, strict=True):
        p = jnp.linspace(z_shift - 5, z_shift + 5, 3)
        ax.plot(p, Efinal + dEdz * (p - z_shift), color="black")

    ax.set_xlabel("z shift [mm]")
    ax.set_ylabel("Final energy [MeV]")
    fig.savefig(artifacts_dir / f"benchmark_3p3_rf_z_tune_{fwdlabel}.png")

# --- Benchmark 4: cooling cell (Tables 4 and 5) ------------------------------

CELL_LENGTH = 800.0 * u.mm
COIL_Z = (100.7 * u.mm, 699.3 * u.mm)  # Table 4; second by symmetry
RF_CENTRES = (211.4 * u.mm, 400.0 * u.mm, 588.6 * u.mm)  # 188.6 mm pitch
APERTURE = 81.6 * u.mm  # beam pipe and iris radius, Table 4
N_FRINGE = 3  # cells modelled up- and downstream for correct fringe overlap
ABSORBER_HALF = 5.0 * u.mm  # half the Table 5 thickness, at each end
RF_PHASE_DEG = 20.0  # Table 4, relative to bunching
N_BEAM = 10_000
N_Z = 201  # 4 mm save spacing
SEED_4 = 20240702

KICK_4 = StochasticKick(
    straggling=energy_loss_kick(landau_energy_loss_sampler),
    scattering=scattering_kick(highland_scattering_sampler),
)


def _cooling_cell_field(phase_deg: float = RF_PHASE_DEG) -> EMTensorField:
    """Coils and three-cell RF.

    Coils alternate polarity with an 800 mm period; N_FRINGE cells are added
    up- and downstream so the overlapping fringe fields at the tracking
    boundaries are correct (section 4).
    """
    frequency = 704.0 * u.MHz
    ref = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    beta = ref.beta()
    phase_turns = phase_deg / 360.0

    components: list[EMTensorField] = []

    coil_pos = SOLENOID
    coil_neg = eqx.tree_at(lambda s: s.jphi, SOLENOID, -SOLENOID.jphi)
    for n in range(-N_FRINGE, N_FRINGE + 1):
        for zc, coil in zip(COIL_Z, (coil_pos, coil_neg), strict=True):
            components.append(
                TransformEMField(
                    transform=Transform.make_translation(z=zc + n * CELL_LENGTH),
                    field=coil,
                )
            )

    for zc in RF_CENTRES:
        tc = zc / (beta * u.c_light)
        phase = -2 * u.pi * ((tc * frequency - 0.25 + phase_turns) % 1.0)
        components.append(
            TransformEMField(
                transform=Transform.make_translation(z=zc),
                field=PillboxCavity(
                    length=183.6 * u.mm,
                    frequency=frequency,
                    E0=30.0 * u.MV / u.m,
                    mode="TM",
                    m=0,
                    n=1,
                    p=0,
                    phase=phase,
                ),
            )
        )

    return SumField(components)


def _make_beam(
    key, n: int, *, momentum, beta_perp, eps_perp, sigma_t, sigma_E
) -> MuonStateDz:
    """Cylindrically symmetric Gaussian beam with alpha = 0 and L_kin = 0.

    Table 4 gives sigma_x = 0.37592 mm, sigma_px = 0.70266 MeV/c.
    Table 5 gives sigma_x = 11.88773 mm, sigma_px = 22.22005 MeV/c.
    """
    ref = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=momentum),
        q=1,
    )
    m, E0 = ref.mass, ref.kin.t.ct

    kx, kpx, ky, kpy, kt, kE = jr.split(key, 6)
    sig_x = jnp.sqrt(beta_perp * eps_perp * m / momentum)
    sig_p = jnp.sqrt(eps_perp * m * momentum / beta_perp)

    x = sig_x * jr.normal(kx, (n,))
    y = sig_x * jr.normal(ky, (n,))
    px = sig_p * jr.normal(kpx, (n,))
    py = sig_p * jr.normal(kpy, (n,))
    E = E0 + sigma_E * jr.normal(kE, (n,))
    ct = u.c_light * sigma_t * jr.normal(kt, (n,))
    pz = jnp.sqrt(E**2 - m**2 - px**2 - py**2)

    # MuonState.make computes E via coords.dot(coords), which only works for a
    # single 3-vector, so build the ensemble one particle at a time under vmap.
    @jax.vmap
    def one(x, y, ct, px, py, pz) -> MuonStateDz:
        return MuonStateDz.make(
            position=Cartesian4.make(x=x, y=y, z=0.0 * u.mm, ct=ct),
            momentum=Cartesian3.make(x=px, y=py, z=pz),
            q=1,
        )

    return one(x, y, ct, px, py, pz)


def _optics(track: MuonStateDz, mask, index: int, mass: float) -> dict[str, float]:
    """Optical quantities at save point `index`, over the masked particles.

    eps_perp = det(Sigma)^(1/4) / m over (x, px, y, py)   [mm]
    beta_perp = p <(x^2 + y^2)/2> / (m eps_perp)          [mm]
    alpha_perp = -(<x px> + <y py>) / (2 m eps_perp)
    L_kin = <x py - y px>                                 [mm MeV/c]
    eps_long = sqrt(det Sigma) over (t, E)                [eV ms, per Tables 4/5]
    """
    sel = np.asarray(mask)
    x = np.asarray(track.kin.p.x[sel, index]) / u.mm
    y = np.asarray(track.kin.p.y[sel, index]) / u.mm
    ct = np.asarray(track.kin.p.ct[sel, index])
    px = np.asarray(track.kin.t.x[sel, index]) / u.MeV
    py = np.asarray(track.kin.t.y[sel, index]) / u.MeV
    pz = np.asarray(track.kin.t.z[sel, index]) / u.MeV
    E = np.asarray(track.kin.t.ct[sel, index]) / u.MeV

    S = np.cov(np.stack([x, px, y, py]))
    eps_perp = float(np.linalg.det(S) ** 0.25) / mass
    p_mean = float(np.mean(np.sqrt(px**2 + py**2 + pz**2)))
    beta_perp = p_mean * 0.5 * (x.var() + y.var()) / (mass * eps_perp)
    alpha_perp = -0.5 * (
        np.cov(x, px)[0, 1] + np.cov(y, py)[0, 1]
    ) / (mass * eps_perp)
    L_kin = float(np.mean(x * py - y * px))

    t_ms = ct / u.c_light / u.ns * 1e-6  # CLHEP ct -> ns -> ms
    E_eV = E * 1e6
    eps_long = float(np.sqrt(np.linalg.det(np.cov(np.stack([t_ms, E_eV])))))

    return {
        "eps_perp_mm": eps_perp,
        "beta_perp_mm": beta_perp,
        "alpha_perp": float(alpha_perp),
        "L_kin_mm_MeV": L_kin,
        "eps_long_eV_ms": eps_long,
        "mean_E_MeV": float(E.mean()),
        "sigma_E_MeV": float(E.std()),
        "p_mean_MeV": p_mean,
    }


def _write_cell_artifacts(artifacts_dir, stem, zs, track, survived, mass, header):
    """Optics vs z, end-plane profiles, and summary plots."""
    rows = [_optics(track, survived, i, mass) for i in range(len(zs))]

    with open(artifacts_dir / f"{stem}_optics.csv", "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        keys = list(rows[0])
        f.write("z_mm," + ",".join(keys) + "\n")
        for i, r in enumerate(rows):
            f.write(
                f"{zs[i] / u.mm:.4f},"
                + ",".join(f"{r[k]:.8g}" for k in keys)
                + "\n"
            )

    sel = np.asarray(survived)
    with open(artifacts_dir / f"{stem}_profiles.csv", "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write("x_mm,px_MeV,y_mm,py_MeV,t_ns,KE_MeV\n")
        m_state = mass
        for j in range(int(sel.sum())):
            f.write(
                f"{float(track.kin.p.x[sel, -1][j]) / u.mm:.6f},"
                f"{float(track.kin.t.x[sel, -1][j]) / u.MeV:.6f},"
                f"{float(track.kin.p.y[sel, -1][j]) / u.mm:.6f},"
                f"{float(track.kin.t.y[sel, -1][j]) / u.MeV:.6f},"
                f"{float(track.kin.p.ct[sel, -1][j]) / u.c_light / u.ns:.6f},"
                f"{float(track.kin.t.ct[sel, -1][j]) / u.MeV - m_state:.6f}\n"
            )

    zmm = np.asarray(zs) / u.mm
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, lab in zip(
        axes.flat,
        ["beta_perp_mm", "L_kin_mm_MeV", "eps_perp_mm", "eps_long_eV_ms"],
        [
            r"$\beta_\perp$ [mm]",
            r"$L_{kin}$ [mm MeV/c]",
            r"$\epsilon_\perp$ [mm]",
            r"$\epsilon_L$ [eV ms]",
        ],
        strict=True,
    ):
        ax.plot(zmm, [r[key] for r in rows], lw=1.2)
        ax.set(xlabel="z [mm]", ylabel=lab)
        ax.grid(alpha=0.3)
    fig.suptitle(stem.replace("_", " "))
    fig.tight_layout()
    fig.savefig(artifacts_dir / f"{stem}_optics.png", dpi=140)
    plt.close(fig)

    return rows


@pytest.mark.extended
def test_benchmark_4_cooling_cell_no_absorber(artifacts_dir):
    """Benchmark 4: low-emittance beam through the cooling cell, no absorber.

    Parameters from Table 4.
    """
    field = _cooling_cell_field()
    ref = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    mass = float(ref.mass) / u.MeV

    beam = _make_beam(
        jr.key(SEED_4),
        N_BEAM,
        momentum=200.0 * u.MeV,
        beta_perp=107.0 * u.mm,
        eps_perp=2.5e-3 * u.mm,
        sigma_t=0.003532 * u.ns,
        sigma_E=0.3692 * u.MeV,
    )
    zs = jnp.linspace(0.0 * u.mm, CELL_LENGTH, N_Z)

    @jax.jit
    @jax.vmap
    def run(start: MuonStateDz) -> MuonStateDz:
        ys, _ = diffrax_solve(field=field, start=start, cts=zs, rtol=1e-7, atol=1e-9)
        return ys

    track = run(beam)

    rho = jnp.hypot(track.kin.p.x, track.kin.p.y)
    survived = np.asarray(jnp.all(rho <= APERTURE, axis=-1))
    transmission = float(survived.mean())

    entry = _optics(track, survived, 0, mass)
    sel = survived
    print(
        f"N={N_BEAM}  "
        f"sigma_x={float(np.std(np.asarray(track.kin.p.x[sel, 0]) / u.mm)):.6f} "
        f"(target 0.375923)  "
        f"sigma_px={float(np.std(np.asarray(track.kin.t.x[sel, 0]) / u.MeV)):.6f} "
        f"(target 0.702660)  "
        f"eps={entry['eps_perp_mm']:.6e}"
    )
    exit_ = _optics(track, survived, -1, mass)

    header = [
        "benchmark 4, Table 4 (low emittance), no absorber",
        f"n_particles={N_BEAM}",
        f"transmission={transmission:.6f}",
        f"rf_phase_deg={RF_PHASE_DEG}",
        f"n_fringe_cells={N_FRINGE}",
        f"aperture_mm={APERTURE / u.mm}",
    ]
    _write_cell_artifacts(
        artifacts_dir,
        "benchmark_4_cell_no_absorber",
        zs,
        track,
        survived,
        mass,
        header,
    )

    # generator/diagnostics cross-check against Table 4
    assert entry["eps_perp_mm"] == pytest.approx(2.5e-3, rel=0.05)
    assert entry["beta_perp_mm"] == pytest.approx(107.0, rel=0.05)
    assert entry["eps_long_eV_ms"] == pytest.approx(1.304e-3, rel=0.05)
    # symplectic transport: emittance is conserved
    assert exit_["eps_perp_mm"] == pytest.approx(entry["eps_perp_mm"], rel=0.01)
    assert exit_["eps_long_eV_ms"] == pytest.approx(
        entry["eps_long_eV_ms"], rel=0.01
    )
    # a flipped coil polarity shows up here and almost nowhere else
    assert abs(exit_["L_kin_mm_MeV"]) < 0.05
    # RF sign convention: 20 deg off bunching must accelerate
    assert exit_["mean_E_MeV"] > entry["mean_E_MeV"]
    assert transmission == pytest.approx(1.0)


@pytest.mark.extended
def test_benchmark_4_cooling_cell_absorber(artifacts_dir):
    """Benchmark 4: nominal-emittance beam through the cooling cell with absorbers.

    Parameters from Table 5.

    Predicted for 10 mm of the PDG LiH entry (rho = 0.82) at 200 MeV/c:
      dE = 1.7145 MeV, X0 = 970.9 mm, x/X0 = 0.0103
      transverse: cooling -0.970%, heating +0.247%, net -0.72%
      longitudinal: sigma_E 11.680 -> 11.688 MeV (+0.065%)
      RF: transit-time factor 0.652, gain 3.687 MeV, net +1.97 MeV per cell
    """
    field = _cooling_cell_field()
    ref = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    mass = float(ref.mass) / u.MeV

    absorbers = SumMaterialVolume(
        components=[
            TransformMaterialVolume(
                transform=Transform.make_translation(z=zc),
                material=AbsorberCylinder(
                    material=MATERIALS["lithium_hydride_LiH"],
                    radius=APERTURE,
                    length=ABSORBER_HALF,
                ),
            )
            for zc in (
                ABSORBER_HALF / 2,
                CELL_LENGTH - ABSORBER_HALF / 2,
            )
        ]
    )

    beam = _make_beam(
        jr.key(SEED_4 + 1),
        N_BEAM,
        momentum=200.0 * u.MeV,
        beta_perp=107.0 * u.mm,
        eps_perp=2.5 * u.mm,
        sigma_t=0.1117 * u.ns,
        sigma_E=11.68 * u.MeV,
    )
    zs = jnp.linspace(0.0 * u.mm, CELL_LENGTH, N_Z)
    keys = jr.split(jr.key(SEED_4 + 2), N_BEAM)

    @jax.jit
    @jax.vmap
    def run(start: MuonStateDz, key) -> MuonStateDz:
        ys, _ = stochastic_solve(field, absorbers, start, zs, key, kick=KICK_4)
        return ys

    track = run(beam, keys)

    rho = jnp.hypot(track.kin.p.x, track.kin.p.y)
    finite = jnp.all(jnp.isfinite(track.kin.t.z), axis=-1)
    survived = np.asarray(jnp.all(rho <= APERTURE, axis=-1) & finite)
    transmission = float(survived.mean())

    entry = _optics(track, survived, 0, mass)
    exit_ = _optics(track, survived, -1, mass)

    header = [
        "benchmark 4, Table 5 (nominal emittance), 5 mm LiH at each end",
        f"material={MATERIALS['lithium_hydride_LiH'].name}",
        f"density_g_cm3="
        f"{MATERIALS['lithium_hydride_LiH'].density / (u.g / u.cm3):.5f}",
        "NOTE density is the PDG tabulated 0.82 g/cm3; benchmark PDF Table 1 "
        "specifies 0.69 for LiH",
        f"absorber_half_thickness_mm={ABSORBER_HALF / u.mm}",
        f"n_particles={N_BEAM}",
        f"transmission={transmission:.6f}",
        f"rf_phase_deg={RF_PHASE_DEG}",
        f"eps_perp_in={entry['eps_perp_mm']:.6f}",
        f"eps_perp_out={exit_['eps_perp_mm']:.6f}",
        f"eps_long_in={entry['eps_long_eV_ms']:.6f}",
        f"eps_long_out={exit_['eps_long_eV_ms']:.6f}",
    ]
    _write_cell_artifacts(
        artifacts_dir,
        "benchmark_4_cell_absorber",
        zs,
        track,
        survived,
        mass,
        header,
    )

    assert transmission > 0.5, f"transmission {transmission:.3f} implausibly low"
    assert exit_["eps_perp_mm"] < entry["eps_perp_mm"], (
        "transverse emittance did not cool; check absorber placement and "
        "RF phase sign"
    )