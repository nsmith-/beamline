import hepunits as u
import jax
import jax.numpy as jnp
import pytest

from beamline.jax.magnet import solenoid as jsol
from beamline.numpy import solenoid as nsol


def test_thin_shell():
    ref = nsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )
    solenoid = jsol.ThinShellSolenoid(R=ref.R, jphi=ref.jphi, L=ref.L)

    rho, z = jnp.meshgrid(
        jnp.linspace(0, 2 * solenoid.R, 100),
        jnp.linspace(-solenoid.L, solenoid.L, 100),
        indexing="ij",
    )
    Brho, Bz = jnp.vectorize(solenoid._B_Caciagli)(rho, z)
    Brho_exp, Bz_exp = ref._B_Caciagli(rho, z)

    assert Brho == pytest.approx(Brho_exp)
    assert Bz == pytest.approx(Bz_exp)


def test_thin_shell_deriv_origin():
    """Test the derivative of the magnetic field for a thin shell solenoid at the origin"""
    sol = jsol.ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
    )

    def bz_origin(sol: jsol.ThinShellSolenoid):
        _, Bz = sol._B_Caciagli(0.0 * sol.R, 0.0 * sol.L)
        return Bz

    def bz_expecetd(sol: jsol.ThinShellSolenoid):
        return jsol.MU0 * sol.jphi * sol.L / 2 / jnp.hypot(sol.R, sol.L / 2)

    val, grad = jax.value_and_grad(bz_origin)(sol)
    val_exp, grad_exp = jax.value_and_grad(bz_expecetd)(sol)

    assert val == pytest.approx(val_exp, rel=1e-8)
    assert grad.R == pytest.approx(grad_exp.R, rel=1e-5)
    assert grad.jphi == pytest.approx(grad_exp.jphi, rel=1e-8)
    assert grad.L == pytest.approx(grad_exp.L, rel=1e-5)
