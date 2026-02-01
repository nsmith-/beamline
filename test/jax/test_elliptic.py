"""Test JAX elliptic integral implementations against numpy/scipy versions"""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.differentiate
import scipy.special
from matplotlib import pyplot as plt

import beamline.jax.elliptic as jell
import beamline.numpy.elliptic as nell

NSAMP = 1000
NSAMP_DERIV = 20


def _gen_samples(case: str, n: int):
    """Helper to generate sample points for testing special cases"""
    rng = np.random.default_rng(1234)
    x = rng.exponential(scale=1.0, size=n)
    y = rng.exponential(scale=1.0, size=n)
    z = rng.exponential(scale=1.0, size=n)
    ops = case.split("_")
    if not all(
        op in ["x0", "y0", "z0", "xeqy", "xeqz", "yeqz", "general"] for op in ops
    ):
        raise ValueError(f"Unknown case: {case}")
    for op in ops:
        if op == "x0":
            x[:] = 0.0
        elif op == "y0":
            y[:] = 0.0
        elif op == "z0":
            z[:] = 0.0
        elif op == "xeqy":
            x[:] = y
        elif op == "xeqz":
            x[:] = z
        elif op == "yeqz":
            y[:] = z
    return x, y, z


def _scipy_ellip_deriv(func: Callable, argnum: int, at: tuple):
    """Helper to compute derivative for carlson integrals (arg >= 0 assumed)"""
    arglist = list(at)
    arg0 = arglist[argnum]

    def wrapper(arg):
        arglist[argnum] = arg
        return func(*arglist)

    out = scipy.differentiate.derivative(
        wrapper,
        arg0,
        initial_step=0.5 if arg0 == 0.0 else min(abs(arg0) / 2, 0.5),
        step_direction=1 if arg0 == 0.0 else 0,
    )
    if not out.success:
        # TODO: investigate
        warnings.warn("derivative computation did not converge", stacklevel=1)
        return pytest.approx(0.0, abs=1e10)
    return pytest.approx(out.df, abs=max(out.error, 1e-13))


@pytest.mark.parametrize("scheme", ["sqrts", "pow"])
def test_powers(benchmark, scheme: str):
    """Checking whether sqrt and integer powers are faster than jnp.pow (yes)"""
    x = np.random.rand(1000).astype(np.float64) + 1.0
    if scheme == "sqrts":
        fun = jax.jit(lambda x: 1 / x**2 / jnp.sqrt(x))
        fun(x)
        benchmark(fun, x)
    else:
        fun = jax.jit(lambda x: jnp.pow(x, -2.5))
        fun(x)
        benchmark(fun, x)


@pytest.mark.parametrize("case", ["general", "x0", "xeqy", "xeqy_z0", "xeqy_yeqz"])
def test_elliprf(case: str):
    x, y, z = _gen_samples(case, n=1000)
    expected = nell.elliprf(x, y, z)
    actual = jax.vmap(jell.elliprf)(x, y, z)
    assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("case", ["general", "x0", "xeqy", "xeqy_z0", "xeqy_yeqz"])
def test_elliprf_deriv(case: str):
    x, y, z = _gen_samples(case, n=100)
    dRdx, dRdy, dRdz = jax.vmap(jax.jacfwd(jell.elliprf, argnums=(0, 1, 2)))(x, y, z)

    for xi, yi, zi, dRdxi, dRdyi, dRdzi in zip(x, y, z, dRdx, dRdy, dRdz, strict=True):
        dRdx_exp = _scipy_ellip_deriv(scipy.special.elliprf, 0, (xi, yi, zi))
        dRdy_exp = _scipy_ellip_deriv(scipy.special.elliprf, 1, (xi, yi, zi))
        dRdz_exp = _scipy_ellip_deriv(scipy.special.elliprf, 2, (xi, yi, zi))
        assert dRdxi == dRdx_exp
        assert dRdyi == dRdy_exp
        assert dRdzi == dRdz_exp


@pytest.mark.parametrize(
    "case", ["general", "x0", "xeqy", "xeqz", "xeqz_y0", "xeqy_yeqz"]
)
def test_elliprd(case: str):
    x, y, z = _gen_samples(case, n=1000)
    expected = nell.elliprd(x, y, z)
    actual = jax.vmap(jell.elliprd)(x, y, z)
    assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    "case", ["general", "x0", "xeqy", "xeqz", "xeqz_y0", "xeqy_yeqz"]
)
def test_elliprd_deriv(case: str):
    x, y, z = _gen_samples(case, n=100)
    dRdx, dRdy, dRdz = jax.vmap(jax.jacfwd(jell.elliprd, argnums=(0, 1, 2)))(x, y, z)

    for xi, yi, zi, dRdxi, dRdyi, dRdzi in zip(x, y, z, dRdx, dRdy, dRdz, strict=True):
        dRdx_exp = _scipy_ellip_deriv(scipy.special.elliprd, 0, (xi, yi, zi))
        dRdy_exp = _scipy_ellip_deriv(scipy.special.elliprd, 1, (xi, yi, zi))
        dRdz_exp = _scipy_ellip_deriv(scipy.special.elliprd, 2, (xi, yi, zi))
        assert dRdxi == dRdx_exp
        assert dRdyi == dRdy_exp
        assert dRdzi == dRdz_exp


@pytest.mark.parametrize("case", ["general", "x0", "xeqy"])
def test_elliprc(case: str):
    x, y, _ = _gen_samples("general", n=1000)
    expected = scipy.special.elliprc(x, y)
    actual = jax.vmap(jell.elliprc)(x, y)
    assert actual == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("case", ["general", "x0", "xeqy"])
def test_elliprc_deriv(case: str):
    x, y, _ = _gen_samples("general", n=100)
    dRdx, dRdy = jax.vmap(jax.jacfwd(jell.elliprc, argnums=(0, 1)))(x, y)

    for xi, yi, dRdxi, dRdyi in zip(x, y, dRdx, dRdy, strict=True):
        dRdx_exp = _scipy_ellip_deriv(scipy.special.elliprc, 0, (xi, yi))
        dRdy_exp = _scipy_ellip_deriv(scipy.special.elliprc, 1, (xi, yi))
        assert dRdxi == dRdx_exp
        assert dRdyi == dRdy_exp


def test_elliprc1p():
    rng = np.random.default_rng(1234)
    ym1 = rng.exponential(scale=1.0, size=NSAMP) - 1.0
    expected = scipy.special.elliprc(1.0, 1.0 + ym1)
    actual = jax.vmap(jell.elliprc1p)(ym1)
    assert actual == pytest.approx(expected, abs=2e-14)


def test_elliprj():
    rng = np.random.default_rng(1234)
    x = rng.exponential(scale=1.0, size=NSAMP)
    y = rng.exponential(scale=1.0, size=NSAMP)
    z = rng.exponential(scale=1.0, size=NSAMP)
    p = rng.exponential(scale=1.0, size=NSAMP)
    # add some (non-colliding) zeros
    x[::8] = 0.0
    y[1::8] = 0.0
    z[2::8] = 0.0

    expected = nell.elliprj(x, y, z, p)
    actual = jax.vmap(jell.elliprj)(x, y, z, p)
    assert actual == pytest.approx(expected, abs=2e-13)


def test_ellipk_asym(artifacts_dir):
    """Test asymptotic expansion of K(k) near k = 1"""
    k = np.geomspace(1e-9, 1e-5, 100)

    K = jax.vmap(lambda kp: jell.elliprf_one_zero(1 - kp**2, 1.0))(k)
    K_exp = scipy.special.ellipk(k**2)
    K_asym = (
        np.pi / 2 + np.pi / 8 * (k**2) / (1 - k**2) - np.pi / 16 * (k**4) / (1 - k**2)
    )

    fig, ax = plt.subplots()
    ax.plot(k, abs(K - np.pi / 2), label="JAX")
    ax.plot(k, abs(K_exp - np.pi / 2), "--", label="scipy")
    ax.plot(k, abs(K_asym - np.pi / 2), ":", label="asymptotic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("abs(K(k) - pi/2)")
    ax.legend()
    fig.savefig(artifacts_dir / "test_ellipk_asym.png")


@pytest.mark.parametrize("case", ["general", "n0", "k0"])
def test_ellipkepi(case: str):
    rng = np.random.default_rng(1234)
    n = rng.uniform(low=-1.0, high=1.0, size=NSAMP)
    k = rng.uniform(low=0.0, high=1.0, size=NSAMP)
    # add some edge cases
    if case == "n0":
        n[:] = 0.0
    elif case == "k0":
        k[:] = 0.0

    exp_K, exp_E, exp_Pi = nell.elliptic_kepi(n, k)
    act_K, act_E, act_Pi = jax.vmap(jell.elliptic_kepi)(n, k)
    assert act_K == pytest.approx(exp_K, abs=2e-14)
    assert act_E == pytest.approx(exp_E, abs=2e-14)
    assert act_Pi == pytest.approx(exp_Pi, abs=1e-13)

    # alternate scipy calculation (note k**2 argument)
    exp_K_alt = scipy.special.ellipk(k**2)
    assert act_K == pytest.approx(exp_K_alt, abs=2e-14)
    exp_E_alt = scipy.special.ellipe(k**2)
    assert act_E == pytest.approx(exp_E_alt, abs=2e-14)


def _scipy_deriv_ellipkepi(n: float, k: float):
    """Compute derivatives of elliptic_kepi using scipy differentiate

    Args:
        n: Parameter for Pi
        k: Modulus

    Returns:
        (dPidn,), (dKdk, dEdk, dPidk): pytest.approx objects for the derivatives
    """
    dPidn = scipy.differentiate.derivative(
        lambda np: nell.elliptic_kepi(np, k)[2],
        n,
        initial_step=min(0.5, abs(1 - n) / 2),
    )

    # initial step default is 0.5 which is too large when k ~ 1
    dKdk = scipy.differentiate.derivative(
        lambda kp: scipy.special.ellipk(kp**2),
        k,
        initial_step=abs(1 - k) / 2,
    )
    dEdk = scipy.differentiate.derivative(
        lambda kp: scipy.special.ellipe(kp**2),
        k,
        initial_step=abs(1 - k) / 2,
    )
    dPidk = scipy.differentiate.derivative(
        lambda kp: nell.elliptic_kepi(n, kp)[2],
        k,
        initial_step=abs(1 - k) / 2,
    )

    return (pytest.approx(dPidn.df, dPidn.error),), (
        pytest.approx(dKdk.df, dKdk.error),
        pytest.approx(dEdk.df, dEdk.error),
        pytest.approx(dPidk.df, dPidk.error),
    )


@pytest.mark.parametrize("case", ["general", "n0", "k0"])
def test_ellipkepi_deriv(case: str):
    rng = np.random.default_rng(1234)
    n = rng.uniform(low=-1.0, high=1.0, size=NSAMP)
    k = rng.uniform(low=0.0, high=1.0, size=NSAMP)
    # add some edge cases
    if case == "n0":
        n[:] = 0.0
    elif case == "k0":
        k[:] = 0.0

    dKdn, dEdn, dPidn = jax.vmap(jax.jacfwd(jell.elliptic_kepi, 0))(n, k)
    dKdk, dEdk, dPidk = jax.vmap(jax.jacfwd(jell.elliptic_kepi, 1))(n, k)

    assert np.all(dKdn == 0.0)
    assert np.all(dEdn == 0.0)

    for ni, ki, dPidni, dKdki, dEdki, dPidki in zip(
        n, k, dPidn, dKdk, dEdk, dPidk, strict=True
    ):
        (dPidn_exp,), (dKdk_exp, dEdk_exp, dPidk_exp) = _scipy_deriv_ellipkepi(ni, ki)
        assert dPidni == dPidn_exp
        assert dKdki == dKdk_exp
        assert dEdki == dEdk_exp
        assert dPidki == dPidk_exp


def test_ellipkepi_deriv_zero():
    """Some limiting cases for the derivatives of elliptic_kepi"""
    # n, k = 0
    dKdn, dEdn, dPidn = jax.jacfwd(jell.elliptic_kepi)(0.0, 0.0)
    assert dKdn == 0.0
    assert dEdn == 0.0
    assert dPidn == 0.0

    dKdk, dEdk, dPidk = jax.jacfwd(jell.elliptic_kepi, argnums=1)(0.0, 0.0)
    assert dKdk == 0.0
    assert dEdk == 0.0
    assert dPidk == 0.0
