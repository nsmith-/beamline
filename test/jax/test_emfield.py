from collections.abc import Callable

import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4, delta_phi
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrators import diffrax_solve
from beamline.jax.kinematics import MuonStateDct
from beamline.jax.types import SFloat


@pytest.mark.parametrize(
    ("Bz", "pxc", "pzc"),
    [
        (1.0 * u.tesla, 10.0 * u.MeV, 0.0 * u.MeV),
        (1.0 * u.tesla, 10.0 * u.MeV, 200.0 * u.MeV),
        (4.0 * u.tesla, 20.0 * u.GeV, 0.0 * u.MeV),
    ],
)
def test_larmor_orbit(artifacts_dir, request, Bz: float, pxc: float, pzc: float):
    """Test Larmor orbit in a uniform magnetic field

    The true path should be a circle in the transverse plane with constant radius
    and uniform angular velocity, and linear motion in the longitudinal direction.
    """
    field = SimpleEMField(
        E0=Cartesian3.make(),
        B0=Cartesian3.make(z=Bz),
    )
    larmor_radius = abs(pxc / u.c_light / Bz)
    start = MuonStateDct.make(
        position=Cartesian4.make(y=larmor_radius),
        momentum=Cartesian3.make(x=pxc, z=pzc),
        q=1,
    )
    larmor_frequency = start.charge * Bz * u.c_light_sq / start.kin.t.ct
    ct0, ct1 = 0.0, 10.0 * u.m
    cts = jnp.linspace(ct0, ct1, 30)

    res = diffrax_solve(field, start, cts)
    res_cyl = res.kin.p.to_cylindric()
    phi = res_cyl.phi
    rho = res_cyl.rho
    z = res_cyl.z

    phi_exp = delta_phi(jnp.pi / 2, (cts - ct0) * larmor_frequency / u.c_light)
    rho_exp = jnp.full_like(cts, larmor_radius)
    betaz = start.kin.t.z / start.kin.t.ct
    z_exp = start.kin.p.z + betaz * (cts - ct0)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))

    ax1.plot(cts, abs(delta_phi(phi, phi_exp)))
    ax2.set_xlabel("ct [mm]")
    ax1.set_ylabel("Angle abs error [rad]")
    ax1.set_yscale("log")

    ax2.plot(cts, abs(rho - rho_exp))
    ax2.set_xlabel("ct [mm]")
    ax2.set_ylabel("Radius abs error [mm]")
    ax2.set_yscale("log")

    ax3.plot(cts, abs(z - z_exp))
    ax3.set_xlabel("ct [mm]")
    ax3.set_ylabel("Longitudinal abs error [mm]")
    ax3.set_yscale("log")

    ax4.plot(res.kin.p.x, res.kin.p.y, label="Numerical")
    ax4.plot(
        rho_exp * jnp.cos(phi_exp), rho_exp * jnp.sin(phi_exp), "--", label="Expected"
    )
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(artifacts_dir / f"{request.node.name}.png")

    # default diffrax_solve tolerances are 1e-5 rel
    assert delta_phi(phi, phi_exp) == pytest.approx(jnp.zeros_like(phi), abs=1e-4)
    assert rho == pytest.approx(rho_exp, rel=1e-4)
    assert z == pytest.approx(z_exp, rel=1e-10)
    assert res_cyl.ct == pytest.approx(cts, rel=1e-10)


def test_diff_solve(artifacts_dir, request):
    pxc = 10.0 * u.MeV
    pzc = 200.0 * u.MeV
    Bz = 1.0 * u.tesla
    larmor_radius = abs(pxc / u.c_light / Bz)
    start = MuonStateDct.make(
        position=Cartesian4.make(y=larmor_radius),
        momentum=Cartesian3.make(x=pxc, z=pzc),
        q=1,
    )
    ct0, ct1 = 0.0, 4.0 * u.m
    cts = jnp.linspace(ct0, ct1, 20)

    def func(B: Cartesian3) -> MuonStateDct:
        field = SimpleEMField(
            E0=Cartesian3.make(),
            B0=B,
        )
        return diffrax_solve(field, start, cts, forward_mode=True)

    Bstart = Cartesian3.make(z=Bz)
    path, dpath_dB = func(Bstart), jax.jacfwd(func)(Bstart)
    """dpath_dB starts as:
    MuonState(
        kin=TangentVector(
            p=Cartesian4(coords=Cartesian3(coords=f64[10, 4, 3])),
            t=Cartesian4(coords=Cartesian3(coords=f64[10, 4, 3])),
        ),
        q=Cartesian3(coords=f64[10, 3]),
    )
    We can make some utility functions to extract the components we want to plot.
    """

    def extract(
        dstate: MuonStateDct, get: Callable[[Cartesian3], SFloat]
    ) -> MuonStateDct:
        return jax.tree.map(
            get,
            dstate,
            is_leaf=lambda x: isinstance(x, Cartesian3) and isinstance(x.coords, Array),
        )

    dpath_dBx = extract(dpath_dB, lambda c: c.x)
    dpath_dBy = extract(dpath_dB, lambda c: c.y)
    dpath_dBz = extract(dpath_dB, lambda c: c.z)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes[0, 0].set_title("d/dBx")
    axes[0, 0].plot(
        path.kin.p.x,
        path.kin.p.y,
        color="grey",
    )
    axes[0, 0].quiver(
        path.kin.p.x,
        path.kin.p.y,
        dpath_dBx.kin.p.x,
        dpath_dBx.kin.p.y,
        angles="xy",
        scale_units="xy",
        color="C0",
    )
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("y [mm]")

    axes[0, 1].set_title("d/dBy")
    axes[0, 1].plot(
        path.kin.p.x,
        path.kin.p.y,
        color="grey",
    )
    axes[0, 1].quiver(
        path.kin.p.x,
        path.kin.p.y,
        dpath_dBy.kin.p.x,
        dpath_dBy.kin.p.y,
        angles="xy",
        scale_units="xy",
        color="C1",
    )
    axes[0, 1].set_xlabel("x [mm]")
    axes[0, 1].set_ylabel("y [mm]")

    axes[1, 0].set_title("d/dBz")
    axes[1, 0].plot(
        path.kin.p.x,
        path.kin.p.y,
        color="grey",
    )
    axes[1, 0].quiver(
        path.kin.p.x,
        path.kin.p.y,
        dpath_dBz.kin.p.x,
        dpath_dBz.kin.p.y,
        angles="xy",
        scale_units="xy",
        color="C2",
    )
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("y [mm]")

    axes[1, 1].plot(
        path.kin.p.z,
        path.kin.p.x,
        color="grey",
    )
    for i, lbl, val in zip(
        range(3),
        ("d/dBx", "d/dBy", "d/dBz"),
        (dpath_dBx, dpath_dBy, dpath_dBz),
        strict=True,
    ):
        axes[1, 1].quiver(
            path.kin.p.z,
            path.kin.p.x,
            val.kin.p.z,
            val.kin.p.x,
            angles="xy",
            scale_units="xy",
            color=f"C{i}",
            label=lbl,
        )
    axes[1, 1].set_title("x-z projection")
    axes[1, 1].set_xlabel("z [mm]")
    axes[1, 1].set_ylabel("x [mm]")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(artifacts_dir / f"{request.node.name}.png")
