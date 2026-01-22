import hepunits as u
import jax
import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt

from beamline.jax.magnet import solenoid as jsol
from beamline.jax.types import SFloat
from beamline.numpy import solenoid as nsol


def test_thin_shell_caciagli():
    ref = nsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )
    solenoid = jsol.ThinShellSolenoid(R=ref.R, jphi=ref.jphi, L=ref.L)

    rho, z = jnp.meshgrid(
        jnp.linspace(0, 2 * solenoid.R, 200),
        jnp.linspace(-solenoid.L, solenoid.L, 101),
        indexing="ij",
    )

    # jax.config.update("jax_debug_nans", True)
    Brho, Bz = jnp.vectorize(solenoid._B_Caciagli)(rho, z)
    Brho_exp, Bz_exp = ref._B_Caciagli(rho, z)

    # actually here we're doing better for small rho than numpy
    assert Brho == pytest.approx(Brho_exp, abs=1e-10)
    assert Bz == pytest.approx(Bz_exp)


def test_thin_shell_rhoexpansion():
    ref = nsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )
    solenoid = jsol.ThinShellSolenoid(R=ref.R, jphi=ref.jphi, L=ref.L)
    rho, z = jnp.meshgrid(
        jnp.linspace(0, 0.5 * solenoid.R, 100),
        jnp.linspace(-solenoid.L, solenoid.L, 100),
        indexing="ij",
    )

    Brho, Bz = jnp.vectorize(solenoid._B_rhoexpansion)(rho, z)
    Brho_exp, Bz_exp = ref._B_rhoexpansion(rho, z)
    assert Brho == pytest.approx(Brho_exp)
    assert Bz == pytest.approx(Bz_exp)


def test_thin_shell_deriv_onaxis():
    """Test the derivative of the magnetic field for a thin shell solenoid at the origin"""
    sol = jsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )

    def bz_onaxis(sol: jsol.ThinShellSolenoid, z: SFloat):
        _, Bz = sol._B(0.0, z)
        return Bz

    def bz_expected(sol: jsol.ThinShellSolenoid, z: SFloat):
        return sol.Bz_onaxis(z)

    zvals = jnp.linspace(-2 * sol.L, 2 * sol.L, 101)
    val, grad = jax.vmap(jax.value_and_grad(bz_onaxis), in_axes=(None, 0))(sol, zvals)
    val_exp, grad_exp = jax.vmap(jax.value_and_grad(bz_expected), in_axes=(None, 0))(
        sol, zvals
    )
    assert val == pytest.approx(val_exp, rel=1e-8)
    assert grad.R == pytest.approx(grad_exp.R, rel=1e-5)
    assert grad.jphi == pytest.approx(grad_exp.jphi, rel=1e-8)
    assert grad.L == pytest.approx(grad_exp.L, rel=1e-5)


def test_optimize_rho0limit(artifacts_dir):
    """How the rho -> 0 limit was optimized

    As rho gets smaller, the Caciagli formula gets less accurate, with a minimum around 1e-6 in this example
    """

    solenoid = jsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )

    zpts = jnp.linspace(-2 * solenoid.L, 2 * solenoid.L, 201)

    @jnp.vectorize
    def maxdiff(rho):
        Bz0 = solenoid.Bz_onaxis(zpts)
        Brho_exp, Bz_exp = jax.vmap(solenoid._B_rhoexpansion, in_axes=(None, 0))(
            rho, zpts
        )
        Brho_cac, Bz_cac = jax.vmap(solenoid._B_Caciagli, in_axes=(None, 0))(rho, zpts)
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
