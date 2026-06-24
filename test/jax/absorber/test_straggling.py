"""Tests for the energy-loss samplers.

Covers the dummy stand-in sampler, the ported Landau value-gradient sampler
(compared against ``scipy.stats.landau`` with a KS test) and the Landau
importance-reweighting formula. Artifacts (figures) are written under
``test_artifacts/``.
"""

import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt
from scipy import stats

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.straggling import (
    _landau_log_weight_ratio,
    _standard_landau,
    _straggling_to_landau,
    dummy_energy_loss_sampler,
    landau_energy_loss_sampler,
    landau_energy_loss_sampler_wg,
)
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.kinematics import MuonStateDz


def landau_energy_loss(E, E_mpv, xi):
    """Landau energy-loss PDF (scipy oracle)

    Parameters
    ----------
    E : float or array_like
        Energy loss random variate
    E_mpv : float
        Most probable energy loss parameter
    xi : float
        Width parameter
    """
    landau_loc = E_mpv + xi * (1 - np.euler_gamma - 0.20005183774398613)
    return stats.landau.pdf(
        E, loc=landau_loc + xi * np.log(np.pi / 2), scale=xi * np.pi / 2
    )


TEST_STRAGGLING_PARAMS = MATERIALS["lithium_hydride_LiH"].straggling_params(
    MuonStateDz.make(
        position=Cartesian4.make(), momentum=Cartesian3.make(z=200.0 * u.MeV), q=1
    ),
    thickness=1.0 * u.cm,
)


def test_dummy_sampler_statistics(artifacts_dir):
    """The dummy sampler is one-sided with the prescribed Bethe-Bloch mean."""
    params = TEST_STRAGGLING_PARAMS
    mean = params.mean_energy_loss
    keys = jr.split(jr.key(0), 50_000)
    dE, log_w = jax.vmap(lambda k: dummy_energy_loss_sampler(params, k))(keys)
    samples = np.asarray(dE)

    assert (samples > 0).all()  # one-sided (energy is only lost)
    assert samples.mean() == pytest.approx(mean, rel=0.03)
    assert np.all(np.asarray(log_w) == 0.0)  # value sampler carries no weight

    fig, ax = plt.subplots()
    ax.hist(samples / u.MeV, bins=80)
    ax.axvline(mean / u.MeV, color="k", ls="--", label="mean")
    ax.set_xlabel("energy loss [MeV]")
    ax.set_ylabel("count")
    ax.set_title("Dummy energy-loss sampler (sum of two exponentials)")
    ax.legend()
    fig.savefig(artifacts_dir / "dummy_sampler_hist.png", dpi=150)
    plt.close(fig)


def test_landau_matches_scipy(artifacts_dir):
    """The Landau sampler reproduces scipy.stats.landau (histogram + KS test)."""
    params = TEST_STRAGGLING_PARAMS

    keys = jr.split(jr.key(0), 50_000)
    dE, log_w = jax.vmap(lambda k: landau_energy_loss_sampler(params, k))(keys)
    samples = np.asarray(dE)
    assert np.all(np.asarray(log_w) == 0.0)

    # The mapped (loc, scale) is the scipy.stats.landau parametrization; compare
    # the empirical distribution to it with a Kolmogorov-Smirnov test.
    loc, scale = (float(v) for v in _straggling_to_landau(params))
    ks = stats.kstest(samples, lambda x: stats.landau.cdf(x, loc=loc, scale=scale))
    assert ks.pvalue > 0.05, f"KS test rejected match: {ks}"

    # Plot the histogram against the oracle PDF over the bulk (the Landau tail is
    # unbounded, so clip the view to a sensible percentile range).
    lo, hi = np.percentile(samples, [1, 95])
    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots()
    ax.hist(samples / u.MeV, bins=60, range=(lo / u.MeV, hi / u.MeV), density=True)
    ax.plot(
        grid / u.MeV,
        landau_energy_loss(grid, params.mode_energy_loss, params.xi) * u.MeV,
        "C1",
        label="scipy.stats.landau",
    )
    ax.axvline(params.mode_energy_loss / u.MeV, color="k", ls="--", label="MPV")
    ax.set_xlabel("energy loss [MeV]")
    ax.set_ylabel("density [1/MeV]")
    ax.set_title(f"Landau sampler vs scipy (KS p={ks.pvalue:.2f})")
    ax.legend()
    fig.savefig(artifacts_dir / "landau_sampler_vs_scipy.png", dpi=150)
    plt.close(fig)


def test_landau_sampler_gradients():
    """Both gradient modes differentiate the sampler in its shape parameters.

    SG (value/pathwise): the mean draw is differentiable, with ``d<dE>/d(mode) = 1``
    exactly (the mode enters ``loc`` additively) and a finite ``xi`` derivative.
    WG (weight/score): the gradient of a bounded observable's self-normalized
    weighted mean flows through the log-weight and is finite. (End-to-end gradients
    through full propagation are exercised elsewhere; here we isolate the sampler.)
    """

    def mk(thickness, pz):
        return MATERIALS["lithium_hydride_LiH"].straggling_params(
            MuonStateDz.make(
                position=Cartesian4.make(),
                momentum=Cartesian3.make(z=pz),
                q=1,
            ),
            thickness=thickness,
        )

    t0 = 1.0 * u.cm
    pz0 = 200.0 * u.MeV

    def mean_dE(params):
        keys = jr.split(jr.key(0), 20_000)
        dE, _ = jax.vmap(lambda k: landau_energy_loss_sampler(params, k))(keys)
        return jnp.mean(dE)

    g_thickness = jax.grad(lambda t: mean_dE(mk(t, pz0)))(t0)
    g_pz = jax.grad(lambda p: mean_dE(mk(t0, p)))(pz0)
    assert jnp.isfinite(g_thickness)
    assert jnp.isfinite(g_pz)

    def wg_weighted_observable(thickness):
        params = mk(thickness, pz0)
        keys = jr.split(jr.key(0), 20_000)
        dE, log_w = jax.vmap(lambda k: landau_energy_loss_sampler_wg(params, k))(keys)
        obs = jnp.exp(-0.5 * dE**2)  # bounded
        weights = jnp.exp(log_w - logsumexp(log_w))
        return jnp.sum(weights * obs)

    g_thickness_wg = jax.grad(wg_weighted_observable)(t0)
    assert jnp.isfinite(g_thickness_wg)


def test_landau_gradient_diagnostics(artifacts_dir):
    """Diagnostic histograms: aux variate, per-sample SG and WG gradients.

    For the SG estimator the per-sample gradient ``d(dE)/d(param)`` varies
    because each draw scales a different ``std_landau`` variate. For the WG
    estimator the per-sample gradient of ``log_w`` is the score function
    ``d log p / d param``, which is heavy-tailed for the Landau.
    """

    def mk(thickness, pz):
        return MATERIALS["lithium_hydride_LiH"].straggling_params(
            MuonStateDz.make(
                position=Cartesian4.make(),
                momentum=Cartesian3.make(z=pz),
                q=1,
            ),
            thickness=thickness,
        )

    t0 = 1.0 * u.cm
    pz0 = 200.0 * u.MeV
    keys = jr.split(jr.key(7), 20_000)

    # --- aux variate ---
    _, aux = jax.vmap(_standard_landau)(keys)
    aux = np.asarray(aux)

    fig, ax = plt.subplots()
    lo, hi = np.percentile(aux, [1, 99])
    ax.hist(aux, bins=60, range=(lo, hi), density=True)
    ax.set_xlabel("aux")
    ax.set_ylabel("density")
    ax.set_title("Standard Landau aux variate (ITS intermediate)")
    fig.savefig(artifacts_dir / "landau_aux_hist.png", dpi=150)
    plt.close(fig)

    # --- per-sample SG gradients ---
    def sg_grads(key):
        def dE_fn(t, pz):
            dE, _ = landau_energy_loss_sampler(mk(t, pz), key)
            return dE

        return jax.grad(dE_fn, argnums=(0, 1))(t0, pz0)

    sg_g_t, sg_g_pz = jax.vmap(sg_grads)(keys)
    sg_g_t = np.asarray(sg_g_t)
    sg_g_pz = np.asarray(sg_g_pz)

    # --- per-sample WG gradients (of log_w = score contribution) ---
    def wg_grads(key):
        def log_w_fn(t, pz):
            _, log_w = landau_energy_loss_sampler_wg(mk(t, pz), key)
            return log_w

        return jax.grad(log_w_fn, argnums=(0, 1))(t0, pz0)

    wg_g_t, wg_g_pz = jax.vmap(wg_grads)(keys)
    wg_g_t = np.asarray(wg_g_t)
    wg_g_pz = np.asarray(wg_g_pz)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    entries = [
        (axes[0, 0], sg_g_t, "SG  ∂(dE)/∂(thickness)", [1, 99]),
        (axes[0, 1], sg_g_pz, "SG  ∂(dE)/∂(pz)", [1, 99]),
        (axes[1, 0], wg_g_t, "WG  ∂(log w)/∂(thickness)", [5, 95]),
        (axes[1, 1], wg_g_pz, "WG  ∂(log w)/∂(pz)", [5, 95]),
    ]
    for ax, data, title, pctiles in entries:
        lo, hi = np.percentile(data, pctiles)
        ax.hist(data, bins=60, range=(lo, hi), density=True)
        ax.axvline(
            np.mean(data),
            color="k",
            ls="--",
            label=f"mean={np.mean(data):.3g}",
        )
        ax.set_title(title)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle("Per-sample gradients: SG (pathwise) and WG (score function)")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "landau_gradient_distributions.png", dpi=150)
    plt.close(fig)


def test_landau_reweight(artifacts_dir):
    """Importance-reweighting one Landau parametrization onto another matches.

    Draw a standard-Landau ensemble, map it to a proposal ``(loc, scale0)``, and
    reweight to a target ``(loc, scale1)``. The reweight formula is the exact joint
    ``(x, aux)`` log-density ratio, so it is an unbiased -- but, for the
    heavy-tailed Landau, high-variance -- importance estimator. We therefore check
    it on a *bounded* observable at large sample size (a full-histogram match in
    the tail is too noisy to assert), plus the trivial identity that reweighting to
    the same parameters leaves the weight at one.
    """
    loc = 1.5 * u.MeV
    scale0, scale1 = 0.165 * u.MeV, 0.15 * u.MeV  # mild mismatch -> usable ESS

    keys = jr.split(jr.key(0), 1_000_000)
    std, aux = jax.vmap(_standard_landau)(keys)
    proposal = std * scale0 + loc
    target = std * scale1 + loc  # direct target draw (paired on the same variate)

    log_w = jax.vmap(
        lambda x, a: _landau_log_weight_ratio(x, a, loc, scale0, loc, scale1)
    )(proposal, aux)
    weights = np.asarray(jnp.exp(log_w))
    proposal = np.asarray(proposal)
    target = np.asarray(target)

    # Identity: reweighting to the same parameters is weight one (log-weight zero).
    log_w_same = jax.vmap(
        lambda x, a: _landau_log_weight_ratio(x, a, loc, scale0, loc, scale0)
    )(np.asarray(std) * scale0 + loc, aux)
    assert np.allclose(np.asarray(log_w_same), 0.0, atol=1e-9)

    # Self-normalized importance estimate of a bounded observable matches the
    # direct estimate at the target parameters.
    def g(x):
        return np.exp(-0.5 * ((x - loc) / (2 * scale1)) ** 2)

    direct = g(target).mean()
    reweighted = np.sum(weights * g(proposal)) / np.sum(weights)
    assert reweighted == pytest.approx(direct, rel=0.07)

    # Plot the two reweight modes the branch implements, both recovering the same
    # target from the proposal: value reweighting (remap the underlying variates,
    # giving exact target draws) and weight reweighting (keep the proposal samples,
    # apply importance weights). The un-reweighted proposal is shown for contrast.
    bins = np.linspace(loc - 2 * scale1, loc + 12 * scale1, 80)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist_proposal, _ = np.histogram(proposal, bins=bins, density=True)
    hist_value, _ = np.histogram(target, bins=bins, density=True)
    hist_weight, _ = np.histogram(proposal, bins=bins, weights=weights, density=True)

    fig, ax = plt.subplots()
    ax.step(
        centers / u.MeV,
        hist_proposal * u.MeV,
        where="mid",
        color="C0",
        alpha=0.6,
        label=f"proposal, no reweight (scale={scale0 / u.MeV:.3f})",
    )
    ax.step(
        centers / u.MeV,
        hist_value * u.MeV,
        where="mid",
        color="C1",
        label=f"value reweighting -> target (scale={scale1 / u.MeV:.3f})",
    )
    ax.step(
        centers / u.MeV,
        hist_weight * u.MeV,
        where="mid",
        color="C2",
        ls="--",
        label="weight reweighting -> target",
    )
    ax.set_xlabel("energy loss [MeV]")
    ax.set_ylabel("density [1/MeV]")
    ax.set_title("Landau reweighting modes (value vs weight)")
    ax.legend()
    fig.savefig(artifacts_dir / "landau_reweight.png", dpi=150)
    plt.close(fig)
