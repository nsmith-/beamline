"""Benchmark 3.3: muon through a TM010 pillbox RF cell.

Table 3 of the cooling code benchmarking note.

The time grid now has 11 points rather than 21 (Table 3 specifies a step 
of 0.1/0.704 ns up to 1/0.704 ns), and the trajectory dump the PDF 
checklist asks for has been added.
"""

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import EMTensorField, TransformEMField
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.rfcavity.pillbox import PillboxCavity
from beamline.jax.types import SFloat


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
        jnp.linspace(0.0 * u.mm, 200.0 * u.mm, 21),
        # changed from 21: Table 3 gives step 0.1/0.704 ns up to 1/0.704 ns,
        # which is 11 points. 21 was half the specified step.
        jnp.linspace(0.0 * u.mm, 1.0 / 0.704 * u.ns * u.c_light, 11),
        indexing="ij",
    )
    X = X.at[0].set(1e-8 * u.mm)
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

    # Full trajectories
    ztrack = jnp.linspace(-500.0 * u.mm, 500.0 * u.mm, 100)

    @jax.jit
    @jax.vmap
    def run_track(start: MuonStateDz) -> MuonStateDz:
        ys, _ = diffrax_solve(
            field=cavity, start=start, cts=ztrack, rtol=1e-7, atol=1e-9
        )
        return ys

    tracks = run_track(starts)  # leaves are (nx * nt, nz); only `ends` is reshaped

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

    # X and CT were built with indexing="ij" then flattened, so the particle
    # at radial index ix and time index it is at ip = ix * nt + it.
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