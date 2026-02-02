from typing import Any

import hepunits as u
import jax.numpy as jnp
import pytest
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4, delta_phi
from beamline.jax.emfield import SimpleEMField, particle_interaction
from beamline.jax.kinematics import MuonState, ParticleState


def propagate(_ct: Any, state: ParticleState, field: SimpleEMField) -> ParticleState:
    """Propagate a particle state through an electromagnetic field for use with diffrax"""
    return particle_interaction(state, field)


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
        E0=Cartesian3(coords=jnp.array([0.0, 0.0, 0.0])),
        B0=Cartesian3(coords=jnp.array([0.0, 0.0, Bz])),
    )
    larmor_radius = abs(pxc / u.c_light / Bz)
    start = MuonState.make(
        position=Cartesian4.make(y=larmor_radius),
        momentum=Cartesian3.make(x=pxc, z=pzc),
        q=1,
    )
    larmor_frequency = start.charge * Bz * u.c_light_sq / start.kin.dx.ct
    ct0, ct1 = 0.0, 10.0 * u.m
    cts = jnp.linspace(ct0, ct1, 30)

    sol = diffeqsolve(
        terms=ODETerm(propagate),
        solver=Dopri5(),
        t0=ct0,
        t1=ct1,
        dt0=1 * u.cm,
        y0=start,
        args=field,
        saveat=SaveAt(ts=cts),
    )
    res: MuonState = sol.ys
    res_cyl = res.kin.point.x.to_cylindrical()
    phi = res_cyl.phi
    rho = res_cyl.rho
    z = res_cyl.z

    phi_exp = delta_phi(jnp.pi / 2, (cts - ct0) * larmor_frequency / u.c_light)
    rho_exp = jnp.full_like(cts, larmor_radius)
    betaz = start.kin.dx.z / start.kin.dx.ct
    z_exp = start.kin.point.x.z + betaz * (cts - ct0)

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

    ax4.plot(res.kin.point.x.x, res.kin.point.x.y, label="Numerical")
    ax4.plot(
        rho_exp * jnp.cos(phi_exp), rho_exp * jnp.sin(phi_exp), "--", label="Expected"
    )
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(artifacts_dir / f"{request.node.name}.png")

    assert phi == pytest.approx(phi_exp, rel=3e-10)
    assert rho == pytest.approx(rho_exp, rel=2e-10)
    assert z == pytest.approx(z_exp, rel=1e-10)
    assert res_cyl.ct == pytest.approx(cts, rel=1e-10)
