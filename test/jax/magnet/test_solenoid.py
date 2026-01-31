from functools import partial

import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt

from beamline.jax.magnet import solenoid as jsol
from beamline.numpy import solenoid as nsol

REF_SOLENOID = nsol.ThinShellSolenoid(
    R=43.81 * u.mm,
    jphi=600 * u.ampere / (0.289 * u.mm),
    L=34.68 * u.mm,
)

# parameters from an example in the muon cooling benchmark
SOLENOID = jsol.ThinShellSolenoid(
    R=43.81 * u.mm,
    jphi=600 * u.ampere / (0.289 * u.mm),
    L=34.68 * u.mm,
)


def _get_bfun(solenoid: jsol.ThinShellSolenoid, fun: str):
    if fun == "Caciagli":
        bfun = solenoid.B_Caciagli
    elif fun.startswith("rhoexpansion"):
        order = int(fun.removeprefix("rhoexpansion"))
        bfun = partial(solenoid.B_rhoexpansion, order=order)
    elif fun == "quadrature":
        bfun = solenoid.B_quadloop
    elif fun == "composite":
        bfun = solenoid.B_composite
    elif fun == "dA":
        bfun = solenoid.B_dA
    else:
        raise RuntimeError
    return jax.jit(jnp.vectorize(bfun))


def test_thin_shell_caciagli():
    ref = REF_SOLENOID
    solenoid = SOLENOID

    rho, z = jnp.meshgrid(
        jnp.linspace(0, 2 * solenoid.R, 200),
        jnp.linspace(-solenoid.L, solenoid.L, 101),
        indexing="ij",
    )

    Brho, Bz = jnp.vectorize(solenoid.B_Caciagli)(rho, z)
    Brho_exp, Bz_exp = ref._B_Caciagli(rho, z)

    # actually here we're doing better for small rho than numpy
    assert Brho == pytest.approx(Brho_exp, abs=1e-10)
    assert Bz == pytest.approx(Bz_exp)


def test_thin_shell_rhoexpansion():
    ref = REF_SOLENOID
    solenoid = SOLENOID
    rho, z = jnp.meshgrid(
        jnp.linspace(0, 0.5 * solenoid.R, 100),
        jnp.linspace(-solenoid.L, solenoid.L, 100),
        indexing="ij",
    )

    Brho, Bz = jnp.vectorize(solenoid.B_rhoexpansion)(rho, z)
    Brho_exp, Bz_exp = ref._B_rhoexpansion(rho, z)
    assert Brho == pytest.approx(Brho_exp)
    assert Bz == pytest.approx(Bz_exp)


@pytest.mark.parametrize(
    ("ref", "cmp", "rholim", "rtol", "atol"),
    [
        # Try to tighten tolerances as much as possible, so this is a good reference
        # rho expansion is very good for small rho
        ("rhoexpansion8", "Caciagli", 0.5, 5e-7, 1e-7),
        # but quickly becomes less accurate near R
        ("rhoexpansion8", "Caciagli", 0.9, 0.034, 3e-3),
        ("rhoexpansion8", "quadrature", 0.5, 5e-7, 1e-7),
        # ("rhoexpansion8", "dA", 0.5, 3e-10),
    ],
)
def test_thin_shell_comparison(
    ref: str, cmp: str, rholim: float, rtol: float, atol: float
):
    """Compare two different solenoid field calculation methods"""
    solenoid = SOLENOID
    B0 = solenoid.Bz_onaxis(0.0)

    rho, z = jnp.meshgrid(
        jnp.linspace(0, rholim * solenoid.R, 200),
        jnp.linspace(-solenoid.L, solenoid.L, 201),
        indexing="ij",
    )

    Brho, Bz = _get_bfun(solenoid, cmp)(rho, z)
    Brho_exp, Bz_exp = _get_bfun(solenoid, ref)(rho, z)
    # Scale to fraction of B0 so atol means something physical
    Brho /= B0
    Bz /= B0
    Brho_exp /= B0
    Bz_exp /= B0
    assert Brho == pytest.approx(Brho_exp, abs=atol, rel=rtol)
    assert Bz == pytest.approx(Bz_exp, abs=atol, rel=rtol)


@pytest.mark.skip("not ready")
def test_B_dA():
    solenoid = SOLENOID

    rho, z = jnp.array(0.0 * solenoid.R), jnp.array(0.4 * solenoid.L)
    A, dA_drho = jax.value_and_grad(solenoid.A, argnums=0)(rho, z)
    dA_dz = jax.grad(solenoid.A, argnums=1)(rho, z)
    assert jnp.isfinite(A)
    assert jnp.isfinite(dA_drho)
    assert jnp.isfinite(dA_dz)
    # Brho, Bz = solenoid.B_dA(rho, z)


@pytest.mark.parametrize("bfun", ["rhoexpansion2", "Caciagli"])
def test_thin_shell_divergence(bfun: str):
    """Test the divergence is zero in the solenoid"""
    sol = SOLENOID

    fun = _get_bfun(sol, bfun)
    atol = 3e-15 if bfun == "Caciagli" else 1e-15

    def rhoBrho(rho, z):
        Brho, _ = fun(rho, z)
        return rho * Brho

    def Bz(rho, z):
        _, Bz = fun(rho, z)
        return Bz

    def divergence(rho, z):
        drhoBrho_drho = jax.grad(rhoBrho, argnums=0)(rho, z)
        dBz_dz = jax.grad(Bz, argnums=1)(rho, z)
        return drhoBrho_drho / rho + dBz_dz

    rng_rho, rng_z = jax.random.split(jax.random.PRNGKey(1234), 2)
    npts = 100
    rhovals = jax.random.chisquare(rng_rho, df=2, shape=(npts,)) * sol.R / 3
    zvals = jax.random.uniform(
        rng_z, shape=(npts,), minval=-2 * sol.L, maxval=2 * sol.L
    )

    divvals = jax.vmap(divergence)(rhovals, zvals)
    assert divvals == pytest.approx(jnp.zeros_like(divvals), abs=atol)


@pytest.mark.parametrize(
    "fun",
    [
        "Caciagli",
        "rhoexpansion1",
        "rhoexpansion2",
        "rhoexpansion4",
        "rhoexpansion8",
        "quadrature",
        "composite",
    ],
)
def test_solenoid_performance(benchmark, fun: str):
    solenoid = SOLENOID
    rho, z = jnp.meshgrid(
        jnp.linspace(0, 2 * solenoid.R, 200),
        jnp.linspace(-2 * solenoid.L, 2 * solenoid.L, 101),
        indexing="ij",
    )

    bfun = _get_bfun(solenoid, fun)

    def run_bfun():
        Brho, Bz = bfun(rho, z)
        (Brho + Bz).block_until_ready()

    # warmup
    run_bfun()
    benchmark(run_bfun)


def test_optimize_rho0limit(artifacts_dir):
    """How the rho -> 0 limit was optimized

    As rho gets smaller, the Caciagli formula gets less accurate, with a minimum around 1e-6 in this example
    """

    solenoid = SOLENOID

    zpts = jnp.linspace(-2 * solenoid.L, 2 * solenoid.L, 201)

    @jnp.vectorize
    def maxdiff(rho):
        Bz0 = solenoid.Bz_onaxis(zpts)
        Brho_exp, Bz_exp = jax.vmap(solenoid.B_rhoexpansion, in_axes=(None, 0))(
            rho, zpts
        )
        Brho_cac, Bz_cac = jax.vmap(solenoid.B_Caciagli, in_axes=(None, 0))(rho, zpts)
        return (
            jnp.max(jnp.abs(Bz_exp - Bz0)),
            jnp.max(jnp.abs(Brho_exp)),
            jnp.max(jnp.abs(Bz_cac - Bz0)),
            jnp.max(jnp.abs(Brho_cac)),
        )

    fig, ax = plt.subplots()

    rhovals = jnp.geomspace(1e-10, 1e-2 * solenoid.R, 100)
    dBz_exp, Brho_exp, dBz_cac, Brho_cac = maxdiff(rhovals)
    ax.plot(
        rhovals / solenoid.R,
        dBz_exp,
        color="C0",
        ls="--",
        label="|Bz expansion - Bz on-axis|",
    )
    ax.plot(
        rhovals / solenoid.R, Brho_exp, color="C0", ls="--", label="|Brho expansion|"
    )
    ax.plot(
        rhovals / solenoid.R, dBz_cac, color="C1", label="|Bz Caciagli - Bz on-axis|"
    )
    ax.plot(rhovals / solenoid.R, Brho_cac, color="C1", label="|Brho Caciagli|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rho / R")
    ax.set_ylabel("Max Difference (kT)")
    ax.legend()
    fig.savefig(artifacts_dir / "optimize_rho0limit.png")
