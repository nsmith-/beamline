"""Benchmark 3.2: muon trajectories through a solenoid.

Table 2 of the cooling code benchmarking note.
"""

from collections.abc import Callable

import diffrax
import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from cooling_common import SOLENOID
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.integrate.propagate import diffrax_solve, particle_interaction
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.types import SFloat


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

    # Full trajectories
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

    # Every second particle for a readable figure
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