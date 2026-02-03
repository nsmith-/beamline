"""Cooling code benchmark tests.

As described in:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf

"""

from collections.abc import Callable

import hepunits as u
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.integrators import diffrax_solve
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.types import SFloat


def test_benchmark_3p2_solenoid(artifacts_dir):
    """Benchmark 3.2: Muon through solenoid

    Parameters from Table 2
    """
    solenoid = ThickSolenoid(
        Rin=250.0 * u.mm,
        Rout=419.3 * u.mm,
        jphi=500.0 * u.A / u.mm**2,
        L=140.0 * u.mm,
    )

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

    end: MuonStateDz = jax.vmap(run, in_axes=(None, 0))(solenoid, xpos)
    # TODO: understand why forward_mode=True fails here (when using jacfwd) at x=0.0
    grad: MuonStateDz = jax.vmap(jax.jacrev(run), in_axes=(None, 0))(solenoid, xpos)

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
        solenoid.Rin * jnp.cos(theta),
        solenoid.Rin * jnp.sin(theta),
        ls="--",
        color="gray",
    )
    ax.plot(
        solenoid.Rout * jnp.cos(theta),
        solenoid.Rout * jnp.sin(theta),
        ls="--",
        color="gray",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Benchmark 3.2: Muon through solenoid")
    ax.set_xlim(-solenoid.Rin, solenoid.Rin)
    ax.set_ylim(-solenoid.Rin, solenoid.Rin)
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
