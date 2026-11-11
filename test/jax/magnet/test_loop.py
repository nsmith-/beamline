import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.magnet.loop import WireLoop
from beamline.units import MU0

LOOP = WireLoop(R=100.0 * u.mm, I=50.0 * u.A)


def _biot_savart(loop: WireLoop, rho: float, z: float, n: int = 20000):
    """Brute-force Biot-Savart (Brho, Bz) at (rho, 0, z) for a loop in the xy plane"""
    R, current = float(loop.R), float(loop.I)
    t = (np.arange(n) + 0.5) * 2 * np.pi / n
    src = np.stack([R * np.cos(t), R * np.sin(t), 0 * t], -1)
    dl = np.stack([-R * np.sin(t), R * np.cos(t), 0 * t], -1) * (2 * np.pi / n)
    r = np.array([rho, 0.0, z]) - src
    B = (
        MU0
        * current
        / (4 * np.pi)
        * np.sum(
            np.cross(dl, r) / np.linalg.norm(r, axis=-1, keepdims=True) ** 3, axis=0
        )
    )
    return B[0], B[2]


@pytest.mark.parametrize(
    ("rho", "z"),
    [(30.0, 20.0), (60.0, -40.0), (150.0, 50.0), (80.0, 5.0), (10.0, 300.0)],
)
def test_biot_savart(rho: float, z: float):
    Brho, Bz = LOOP.B(rho * u.mm, z * u.mm)
    Brho_ref, Bz_ref = _biot_savart(LOOP, rho * u.mm, z * u.mm)
    scale = np.hypot(Brho_ref, Bz_ref)
    assert float(Brho) == pytest.approx(Brho_ref, abs=1e-12 * scale)
    assert float(Bz) == pytest.approx(Bz_ref, abs=1e-12 * scale)


def test_onaxis():
    zpts = jnp.linspace(-3 * LOOP.R, 3 * LOOP.R, 101)
    _, Bz = jax.vmap(LOOP.B, in_axes=(None, 0))(1e-3 * LOOP.R, zpts)
    assert Bz == pytest.approx(LOOP.Bz_onaxis(zpts), rel=1e-6)
    # center of a loop: mu0 I / 2R
    assert float(LOOP.Bz_onaxis(0.0)) == pytest.approx(MU0 * LOOP.I / (2 * LOOP.R))


def test_optimize_rho0limit(artifacts_dir):
    """Accuracy of the elliptic formula as rho -> 0, compared to the series expansion

    Mirrors test_solenoid.py::test_optimize_rho0limit
    """
    loop = LOOP

    zpts = jnp.linspace(-2 * loop.R, 2 * loop.R, 201)

    @jnp.vectorize
    def maxdiff(rho):
        Bz0 = loop.Bz_onaxis(zpts)
        Brho_exp, Bz_exp = jax.vmap(loop.B_rhoexpansion, in_axes=(None, 0))(rho, zpts)
        Brho_ell, Bz_ell = jax.vmap(loop.B, in_axes=(None, 0))(rho, zpts)
        return (
            jnp.max(jnp.abs(Bz_exp - Bz0)),
            jnp.max(jnp.abs(Brho_exp)),
            jnp.max(jnp.abs(Bz_ell - Bz0)),
            jnp.max(jnp.abs(Brho_ell)),
        )

    fig, ax = plt.subplots()

    rhovals = jnp.geomspace(1e-16, 1e-6 * loop.R, 100)
    dBz_exp, Brho_exp, dBz_ell, Brho_ell = maxdiff(rhovals)
    _, _, Bz0_ell, Brho0_ell = maxdiff(jnp.array(0.0))
    ax.plot(
        rhovals / loop.R,
        dBz_exp,
        color="C0",
        ls="--",
        label="|Bz expansion - Bz on-axis|",
    )
    ax.plot(rhovals / loop.R, Brho_exp, color="C1", ls="--", label="|Brho expansion|")
    ax.plot(rhovals / loop.R, dBz_ell, color="C0", label="|Bz elliptic - Bz on-axis|")
    ax.plot(rhovals / loop.R, Brho_ell, color="C1", label="|Brho elliptic|")
    ax.axhline(Bz0_ell, color="C0", ls=":", label="|Bz diff @rho=0|")
    ax.axhline(Brho0_ell, color="C1", ls=":", label=f"|Brho={Brho0_ell} @rho=0|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rho / R")
    ax.set_ylabel("Max Difference (kT)")
    ax.legend()
    fig.savefig(artifacts_dir / "optimize_rho0limit_loop.png")


def test_onaxis_exact():
    """Exactly on axis: Brho vanishes, Bz is the closed form, and gradients are finite"""
    zpts = jnp.linspace(-3 * LOOP.R, 3 * LOOP.R, 31)
    Brho, Bz = jax.vmap(LOOP.B, in_axes=(None, 0))(0.0, zpts)
    # Rf - 2/3 Rd cancels to rounding error at rho = 0
    assert jnp.max(jnp.abs(Brho)) < 1e-15 * jnp.max(jnp.abs(Bz))
    assert Bz == pytest.approx(LOOP.Bz_onaxis(zpts), rel=1e-12)

    # derivatives of B(rho, z) are finite on axis, and div B = 0 there:
    # (1/rho) d(rho Brho)/drho -> 2 dBrho/drho as rho -> 0
    J = jax.jacfwd(lambda a: jnp.stack(LOOP.B(a[0], a[1])))(jnp.array([0.0, 30.0]))
    assert jnp.all(jnp.isfinite(J))
    div = 2 * J[0, 0] + J[1, 1]
    assert float(div) == pytest.approx(0.0, abs=1e-12 * float(jnp.abs(J).max()))


@pytest.mark.parametrize("d", [0.0, 1e-14, 1e-10, 1e-6, 1e-2])
def test_near_wire(d: float):
    """Evaluating at or near the wire terminates with a finite value

    The field is not meant to be accurate there (see WireLoop.B)
    """
    R = float(LOOP.R)
    for rho, z in [(R + d, 0.0), (R - d, 0.0), (R, d)]:
        Brho, Bz = LOOP.B(rho, z)
        assert jnp.isfinite(Brho)
        assert jnp.isfinite(Bz)
