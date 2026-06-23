"""Sampling of stochastic energy loss

This module holds the energy-loss *sampler*: a callable
``(StragglingParams, key) -> dE`` mapping the straggling shape parameters and a
PRNG key to an energy loss [MeV]. The real Landau/Vavilov sampler will be
imported here later and can be dropped in wherever a sampler is expected (e.g.
``beamline.jax.integrate.stochastic.stochastic_solve``).

For now ``dummy_energy_loss_sampler`` is a crude, one-sided stand-in used to
exercise the scaffolding: the sum of two exponentials whose *total mean* equals
the Bethe-Bloch mean energy loss. It is sampled by the reparameterization trick
(``Exp(mean) = -mean * log U``) so that the draw is pathwise-differentiable in
``mean_energy_loss``.
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from beamline.jax.absorber.material import StragglingParams
from beamline.jax.types import SFloat


def dummy_energy_loss_sampler(params: StragglingParams, key: Array) -> SFloat:
    """Sample a (dummy) energy loss [MeV] for the given straggling parameters

    Placeholder distribution: the sum of two i.i.d. exponentials each with mean
    ``mean_energy_loss / 2``, so the total has mean ``mean_energy_loss``. The
    draw is one-sided (always positive) and differentiable in
    ``mean_energy_loss``. It does **not** reproduce the Landau/Vavilov shape; it
    is only a stand-in until the real sampler is wired in.

    Args:
        params: Straggling shape parameters (only ``mean_energy_loss`` is used).
        key: A JAX PRNG key.

    Returns:
        The sampled energy loss [MeV].
    """
    k1, k2 = jr.split(key)
    mean_each = params.mean_energy_loss / 2
    # clamp the uniforms away from 0 so log is finite (jax_debug_nans is on)
    tiny = jnp.finfo(jnp.result_type(float)).tiny
    u1 = jr.uniform(k1, minval=tiny, maxval=1.0)
    u2 = jr.uniform(k2, minval=tiny, maxval=1.0)
    return -mean_each * (jnp.log(u1) + jnp.log(u2))
