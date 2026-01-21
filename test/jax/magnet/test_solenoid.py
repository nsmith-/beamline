import equinox as eqx
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


def test_thin_shell_deriv():
    sol = jsol.ThinShellSolenoid(
        R=40 * u.mm,
        jphi=600 * u.ampere / (0.3 * u.mm),
        L=30 * u.mm,
    )

    def f(sol: jsol.ThinShellSolenoid):
        _, Bz = sol._B_Caciagli(0.5 * sol.R, 0.5 * sol.L)
        return Bz

    val, grad = jax.value_and_grad(f)(sol)

    assert val == pytest.approx(0.0008128264759033163)
    raise AssertionError(eqx.tree_pformat(grad, short_arrays=False))
