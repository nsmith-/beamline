"""Test JAX elliptic integral implementations against numpy/scipy versions"""

import jax
import numpy as np
import pytest
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


def test_ellipkepi():
    rng = np.random.default_rng(1234)
    n = rng.uniform(low=-1.0 + np.finfo(np.float64).eps, high=1.0, size=NSAMP)
    k = rng.uniform(low=0.0, high=1.0, size=NSAMP)

    exp_K, exp_E, exp_Pi = nell.elliptic_kepi(n, k)
    act_K, act_E, act_Pi = jax.vmap(jell.elliptic_kepi)(n, k)
    assert act_K == pytest.approx(exp_K, abs=2e-14)
    assert act_E == pytest.approx(exp_E, abs=2e-14)
    assert act_Pi == pytest.approx(exp_Pi, abs=1e-13)


def test_ellipkepi_deriv():
    rng = np.random.default_rng(1234)
    n = rng.uniform(low=-1.0 + np.finfo(np.float64).eps, high=1.0, size=NSAMP)
    k = rng.uniform(low=0.0, high=1.0, size=NSAMP)

    # TODO: does this test anything meaningful?
    fwd_K, fwd_E, fwd_Pi = jax.vmap(jax.jacfwd(jell.elliptic_kepi))(n, k)
    rev_K, rev_E, rev_Pi = jax.vmap(jax.jacrev(jell.elliptic_kepi))(n, k)
    assert fwd_K == pytest.approx(rev_K, abs=1e-13)
    assert fwd_E == pytest.approx(rev_E, abs=1e-13)
    assert fwd_Pi == pytest.approx(rev_Pi, abs=1e-13)
