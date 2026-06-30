"""
test_landau_peak.py
===================

Validates the constant ``_LANDAU_STD_PEAK = -0.42931453`` used in
``absorber.py``. This number is the location of the peak of the
standardized Landau density. ``absorber.py`` needs it to shift the
``loc`` parameter of the in-house Landau sampler so the sampled
distribution peaks at the physical MPV (``mode_energy_loss``) rather
than 0.42931·xi below it.

The proof proceeds in two airtight steps:

  STEP A (analytic):  Maximize scipy.stats.landau.pdf numerically.
                       The result, -0.42931453, is a property of the
                       density definition; no Monte Carlo involved.

  STEP B (statistical): Show that the in-house C-M-S sampler in
                       diff_random.distributions._landau.Landau_SG
                       samples from the SAME density as
                       scipy.stats.landau. We do this by binning the
                       sampler's output and chi^2-comparing to the
                       analytic CDF differences of scipy.stats.landau.

If both steps pass, the in-house sampler's mode IS scipy's PDF mode
IS -0.42931453, and the constant in absorber.py is provably correct.

A third test (STEP C) is a loose sanity check that the empirical peak
of the in-house sampler lands near -0.42931 — useful for human
inspection, not part of the proof (extracting the mode of a Landau
distribution from samples is noisy because the peak is very flat).

Run from the repo root:

    uv run pytest test/jax/test_landau_peak.py -v -s
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import landau
from scipy.optimize import minimize_scalar


# The number we are validating. Derived analytically by test_pdf_peak.
EXPECTED_PEAK = -0.42931453


# ---------------------------------------------------------------------------
# STEP A — derive the constant from the analytic PDF
# ---------------------------------------------------------------------------
def test_pdf_peak_equals_constant():
    """The peak of scipy.stats.landau.pdf(x, loc=0, scale=1) IS the
    convention. Find it by Brent maximization. The result must match
    the constant used in absorber.py to high precision.
    """
    res = minimize_scalar(
        lambda x: -landau.pdf(x),
        bracket=(-2.0, -0.4, 2.0),
        method="brent",
        options={"xtol": 1e-12},
    )
    pdf_peak = float(res.x)
    print(f"\n  scipy.stats.landau PDF peak: {pdf_peak:.8f}")
    print(f"  absorber.py constant:        {EXPECTED_PEAK:.8f}")
    print(f"  difference:                  {pdf_peak - EXPECTED_PEAK:+.2e}")
    assert abs(pdf_peak - EXPECTED_PEAK) < 1e-7, (
        f"PDF peak {pdf_peak} disagrees with absorber.py constant "
        f"{EXPECTED_PEAK}. Update _LANDAU_STD_PEAK."
    )


# ---------------------------------------------------------------------------
# STEP B — prove the in-house sampler matches scipy's density
# ---------------------------------------------------------------------------
def test_diff_random_density_matches_scipy_landau():
    """Bin samples from diff_random.distributions._landau.Landau_SG and
    chi^2-compare against (CDF differences of) scipy.stats.landau.

    Passing this test proves the in-house sampler and scipy.stats.landau
    sample from the *same* distribution. Combined with test_pdf_peak,
    this proves the in-house sampler's mode is at -0.42931453.
    """
    import jax
    from diff_random.distributions._landau import Landau_SG

    N = 2_000_000
    key = jax.random.key(42)
    keys = jax.random.split(key, N)
    dist = Landau_SG(loc=0.0, scale=1.0)
    samples = np.asarray(
        jax.vmap(lambda k: dist._generate_one_sample(k)[0])(keys)
    )

    edges = np.linspace(-3.0, 8.0, 110)
    obs, _ = np.histogram(samples, bins=edges)
    cdf = landau.cdf(edges)
    expected = (cdf[1:] - cdf[:-1]) * N

    # Pearson chi^2 over bins with enough expected counts (>=50 to keep
    # the asymptotic chi^2 approximation valid).
    mask = expected >= 50
    chi2 = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = int(mask.sum() - 1)
    chi2_per_dof = chi2 / dof
    print(f"\n  chi^2 = {chi2:.1f}, dof = {dof}, chi^2/dof = {chi2_per_dof:.2f}")
    print(f"  (chi^2/dof near 1 means the two samplers draw from the same density)")

    # For dof ~ 100 the 99% upper limit on chi^2/dof is about 1.4.
    # 2.0 leaves headroom for unrelated noise without admitting real
    # distributional differences.
    assert chi2_per_dof < 2.0, (
        f"diff_random sampler does NOT match scipy.stats.landau "
        f"(chi^2/dof = {chi2_per_dof:.2f}). The constant "
        f"_LANDAU_STD_PEAK = {EXPECTED_PEAK} would be wrong because "
        f"it was derived assuming the two samplers agree."
    )


# ---------------------------------------------------------------------------
# STEP C — empirical sanity check (optional, noisy)
# ---------------------------------------------------------------------------
def test_diff_random_empirical_peak_sanity():
    """Loose sanity check: empirical peak of diff_random samples lies
    near -0.42931. Uses a quadratic fit over the peak region (which is
    unbiased, unlike a Gaussian fit). Tolerance is loose because the
    Landau peak is very flat and the mode is hard to localize from
    samples alone.

    This test is NOT part of the proof — the proof is tests A and B.
    But it provides a human-readable cross-check.
    """
    import jax
    from diff_random.distributions._landau import Landau_SG

    N = 5_000_000
    key = jax.random.key(20260617)
    keys = jax.random.split(key, N)
    dist = Landau_SG(loc=0.0, scale=1.0)
    samples = np.asarray(
        jax.vmap(lambda k: dist._generate_one_sample(k)[0])(keys)
    )

    # Quadratic fit on a wide window around the peak (the Landau peak
    # is so flat that a fine-binned argmax just picks up Poisson noise;
    # a quadratic averages the curvature over a wider region).
    bw = 0.02
    edges = np.arange(-2.0, 1.0 + bw, bw)
    h, _ = np.histogram(samples, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # smoothed argmax for coarse peak location
    k = 31
    hs = np.convolve(h.astype(float), np.ones(k) / k, mode="same")
    i0 = int(hs.argmax())
    lo, hi = max(0, i0 - k), min(len(centers), i0 + k + 1)
    x, y = centers[lo:hi], h[lo:hi].astype(float)
    w = 1.0 / np.maximum(y, 1.0)
    a, b, _ = np.polyfit(x, y, 2, w=w)
    mu = -b / (2 * a)

    print(f"\n  diff_random empirical peak (quadratic fit): {mu:.4f}")
    print(f"  expected (from PDF maximization):           {EXPECTED_PEAK:.4f}")
    print(f"  (tolerance is loose; rigorous proof is in test_diff_random_density_matches_scipy_landau)")

    # Generous tolerance — the Landau peak is genuinely hard to localize
    # from samples. A failure at this level (> 0.05) would indicate the
    # sampler is in a completely different convention, which the chi^2
    # test would have already caught.
    assert abs(mu - EXPECTED_PEAK) < 0.05, (
        f"diff_random empirical peak {mu:.4f} is wildly off from "
        f"{EXPECTED_PEAK}. Check the chi^2 test for a more rigorous diagnosis."
    )


if __name__ == "__main__":
    test_pdf_peak_equals_constant()
    print("\n[A] PDF maximum confirms _LANDAU_STD_PEAK = -0.42931453")
    try:
        test_diff_random_density_matches_scipy_landau()
        print("[B] in-house sampler matches scipy.stats.landau density")
        test_diff_random_empirical_peak_sanity()
        print("[C] empirical peak sanity check passes")
        print("\nAll tests passed. The constant is validated.")
    except ImportError:
        print("[B,C] skipped — diff_random not importable in this environment")
