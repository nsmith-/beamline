"""Test JAX elliptic integral implementations against numpy/scipy versions"""

import jax
import numpy as np
import pytest
import scipy.differentiate
import scipy.special

import beamline.jax.elliptic as jell
import beamline.numpy.elliptic as nell

NSAMP = 1000


def test_elliprf():
    rng = np.random.default_rng(1234)
    x = rng.exponential(scale=1.0, size=NSAMP)
    y = rng.exponential(scale=1.0, size=NSAMP)
    z = rng.exponential(scale=1.0, size=NSAMP)
    # add some (non-colliding) zeros
    x[::8] = 0.0
    y[1::8] = 0.0
    z[2::8] = 0.0

    expected = nell.elliprf(x, y, z)
    actual = jax.vmap(jell.elliprf)(x, y, z)
    assert actual == pytest.approx(expected, abs=1e-15)


def test_elliprd():
    rng = np.random.default_rng(1234)
    x = rng.exponential(scale=1.0, size=NSAMP)
    y = rng.exponential(scale=1.0, size=NSAMP)
    z = rng.exponential(scale=1.0, size=NSAMP)
    # add some (non-colliding) zeros
    x[::8] = 0.0
    y[1::8] = 0.0

    expected = nell.elliprd(x, y, z)
    actual = jax.vmap(jell.elliprd)(x, y, z)
    # TODO: investigate why abs tol needs to be looser here
    assert actual == pytest.approx(expected, abs=1e-13)


def test_elliprc():
    rng = np.random.default_rng(1234)
    x = rng.exponential(scale=1.0, size=NSAMP)
    y = rng.exponential(scale=1.0, size=NSAMP)
    # add some zeros
    x[::8] = 0.0

    expected = scipy.special.elliprc(x, y)
    actual = jax.vmap(jell.elliprc)(x, y)
    assert actual == pytest.approx(expected, abs=2e-14)


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
