"""Test JAX Bessel function implementations against numpy/scipy versions"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import scipy.special
from matplotlib import pyplot as plt

import beamline.jax.bessel as jbessel

NSAMP = 10_000


def test_plot_bessel(artifacts_dir: Path):
    x = jnp.geomspace(1e-10, 1e3, 1000)

    fig, ax = plt.subplots()

    for v in [0, 1, 2, 3, 4]:
        expected = scipy.special.jv(v, x)
        computed = jax.vmap(jbessel.jv, in_axes=(None, 0))(v, x)
        ax.plot(x, abs(expected - computed), ".", label=f"{v=}")

    ax.axhline(jnp.finfo("d").eps, color="gray", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 1e-12)
    ax.set_xlabel("x")
    ax.set_ylabel("|scipy - jax|")
    ax.legend(title=r"Bessel $J_v(x)$")
    fig.savefig(artifacts_dir / "bessel_jv_difference.png")


# We don't need very large orders for RF cavities
@pytest.mark.parametrize("v", range(-2, 5))
def test_jax_bessel_jv(v: int):
    x = jnp.geomspace(1e-10, 1e3, NSAMP)
    expected = scipy.special.jv(v, x)
    actual = jax.vmap(jbessel.jv, in_axes=(None, 0))(v, x)
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("v", range(5))
def test_jax_bessel_jvprime(v: int):
    x = jnp.geomspace(1e-10, 1e3, NSAMP)
    expected = scipy.special.jvp(v, x)
    jv_func = partial(jbessel.jv, v)
    actual = jax.vmap(jax.grad(jv_func))(x)
    assert actual == pytest.approx(expected, abs=3e-14)


@pytest.mark.parametrize("v", [1, 2])
def test_jv_over_z(artifacts_dir, v: int):
    """Study of jv(z)/z

    Unlike in numpy case, this seems to be stable at small z
    """

    cutoff = 1e-8
    z = jnp.geomspace(1e-4 * cutoff, 1e1 * cutoff, 100)

    jvz = jax.vmap(jbessel.jv, in_axes=(None, 0))(v, z) / z
    asym = jnp.pow(z, v - 1) / (2**v * jbessel.gamma(v + 1))

    fig, ax = plt.subplots()
    ax.plot(z, jvz - asym, label="jv(z)/z - asym")
    ax.legend()
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / f"jv_over_z_v{v}.png")
