from pathlib import Path

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt

from beamline.jax.emfield import StackedField, TransformEMField

pytest.importorskip("pxr", reason="usd-core not installed")


from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.export.usd import add_trajectories, add_volume, make_stage, save_usdz
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDct
from beamline.jax.magnet.loop import WireLoop

Z_PERIOD = 3.0 * u.m
"""The period of the helix along the z-axis."""
COIL_RADIUS = 100.0 * u.mm
"""The radius of the coils."""
COIL_CURRENT = 100.0 * u.A
"""The current through the coils."""


def test_helmholz_helix(artifacts_dir: Path):
    """A double-helix like structure of helmholz coils

    The goal is to see if this can match the dipole field pattern
    of the helical FOFO lattice (which nominally uses tilted solenoids)
    """
    path = str(artifacts_dir / "helmholz_helix.usdz")
    stage = make_stage(path)

    def place(z: jax.Array, shift: jax.Array):
        phi = z * 2 * jnp.pi / Z_PERIOD
        return TransformEMField(
            transform=Transform.make_axis_angle(
                axis=Cartesian3.make(x=jnp.cos(phi), y=jnp.sin(phi)),
                angle=jnp.pi / 2,
                translation=Cartesian4.make(
                    x=shift * jnp.sin(phi), y=-shift * jnp.cos(phi), z=z
                ),
            ),
            field=WireLoop(
                R=COIL_RADIUS,
                I=COIL_CURRENT,
            ),
        )

    zs, shifts = jnp.meshgrid(
        jnp.linspace(0.0 * u.m, Z_PERIOD, 10),
        jnp.array([COIL_RADIUS / 2, -COIL_RADIUS / 2]),
        indexing="ij",
    )
    loops = StackedField(fields=eqx.filter_vmap(place)(zs.ravel(), shifts.ravel()))

    add_volume(stage, "/beamline/loop", loops)

    field = loops
    start = MuonStateDct.make(
        position=Cartesian4.make(z=-1.0 * u.m),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    cts = jnp.linspace(0.0, 2 * Z_PERIOD, 100)
    states, _ = diffrax_solve(field, start, cts)
    add_trajectories(stage, "/trajectories", states, animate_marker=True)

    save_usdz(stage)

    # plot radial field magnitude and phase along the z-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    zs = jnp.linspace(0.0 * u.m, Z_PERIOD, 500)
    _, B = jax.vmap(field.field_strength)(Cartesian4.make(z=zs))

    fig.suptitle("Helmholz Helix Field")
    ax1.plot(zs / u.m, jnp.hypot(B.t.x, B.t.y) / u.T)
    ax2.plot(zs / u.m, jnp.arctan2(B.t.y, B.t.x))
    ax1.set_ylabel("Transverse B [T]")
    ax2.set_ylabel("Phase")
    ax2.set_xlabel("z [m]")
    fig.savefig(f"{artifacts_dir}/helmholz_helix_field.png")
