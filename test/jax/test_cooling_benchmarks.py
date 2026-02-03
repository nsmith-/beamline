"""Cooling code benchmark tests.

As described in:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf

"""

from collections.abc import Callable

import diffrax
import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.integrators import diffrax_solve, propagate
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.types import SFloat

SOLENOID = ThickSolenoid(
    Rin=250.0 * u.mm,
    Rout=419.3 * u.mm,
    jphi=500.0 * u.A / u.mm**2,
    L=140.0 * u.mm,
)


def test_benchmark_3p2_solenoid(artifacts_dir):
    """Benchmark 3.2: Muon through solenoid

    Parameters from Table 2
    """
    xpos = jnp.arange(-200.0 * u.mm, 201.0 * u.mm, 10.0 * u.mm)
    z_span = (-500.0 * u.mm, 500.0 * u.mm)

    @jax.jit
    def run(fieldobj: ThickSolenoid, xstart: SFloat) -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(x=xstart, z=-500.0 * u.mm),
            momentum=Cartesian3.make(z=200 * u.MeV),
            q=1,
        )
        zs = jnp.array(z_span)
        sol = diffrax_solve(fieldobj, start, zs, forward_mode=False)
        return jax.tree.map(lambda x: x[-1], sol)

    end: MuonStateDz = jax.vmap(run, in_axes=(None, 0))(SOLENOID, xpos)
    # TODO: understand why forward_mode=True fails here (when using jacfwd) at x=0.0
    grad: MuonStateDz = jax.vmap(jax.jacrev(run), in_axes=(None, 0))(SOLENOID, xpos)

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
            end.kin.point.x.x / u.mm,
            end.kin.point.x.y / u.mm,
            end.kin.point.x.z / u.mm,
            end.kin.point.x.ct / u.mm,
            end.kin.dx.x / u.MeV,
            end.kin.dx.y / u.MeV,
            end.kin.dx.z / u.MeV,
            end.kin.dx.ct / u.MeV,
        ]
        for row in zip(*cols, strict=True):
            f.write(",".join(f"{val:.6f}" for val in row) + "\n")

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        end.kin.point.x.x,
        end.kin.point.x.y,
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
    ax.set_title("Benchmark 3.2: Muon through solenoid")
    ax.set_xlim(-SOLENOID.Rin, SOLENOID.Rin)
    ax.set_ylim(-SOLENOID.Rin, SOLENOID.Rin)
    ax.legend(loc="upper center")

    # ax.set_facecolor("none")
    # fig.set_facecolor("none")
    fig.savefig(artifacts_dir / "benchmark_3p2_solenoid.png", dpi=150)

    scale = 0.5
    ax.quiver(
        end.kin.point.x.x,
        end.kin.point.x.y,
        grad_Rin.kin.point.x.x,
        grad_Rin.kin.point.x.y,
        color="blue",
        label="d/dRin",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    ax.quiver(
        end.kin.point.x.x,
        end.kin.point.x.y,
        grad_Rout.kin.point.x.x,
        grad_Rout.kin.point.x.y,
        color="orange",
        label="d/dRout",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    ax.quiver(
        end.kin.point.x.x,
        end.kin.point.x.y,
        grad_jphi.kin.point.x.x * (u.A / u.mm**2),
        grad_jphi.kin.point.x.x * (u.A / u.mm**2),
        color="green",
        label="d/djphi",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    ax.quiver(
        end.kin.point.x.x,
        end.kin.point.x.y,
        grad_L.kin.point.x.x,
        grad_L.kin.point.x.y,
        color="red",
        label="d/dL",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=3e-3,
    )
    ax.legend(loc="upper center")
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
            terms=diffrax.ODETerm(propagate),
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
            terms=diffrax.ODETerm(propagate),
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
            lambda x, y: jnp.maximum(x, y),
            jax.tree.map(
                lambda x, y: leaf_func(x, y),
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
            result.kin.point,
            reference_benchmark_3p2_solenoid.kin.point,
        ),
        "max_abs_diff_mom": reduce(
            abs_diff,
            result.kin.dx,
            reference_benchmark_3p2_solenoid.kin.dx,
        ),
        "max_rel_diff_pos": reduce(
            rel_diff,
            result.kin.point,
            reference_benchmark_3p2_solenoid.kin.point,
        ),
        "max_rel_diff_mom": reduce(
            rel_diff,
            result.kin.dx,
            reference_benchmark_3p2_solenoid.kin.dx,
        ),
    }
    benchmark(runbench)
