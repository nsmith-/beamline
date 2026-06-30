"""
dist_stats.py
=============

spectrum: the *mode* via a Gaussian fit to the peak, and the *mean* / *median*

Only numpy + scipy are required.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_peak_fit(samples, bins=400, k=1.0, n_iter=4, init_window_frac=0.5,
                      hist_range=None):
    """Iterative Gaussian fit to the PEAK of a skewed distribution.

    Parameters
    ----------
    samples : 1-D array of energy-loss values.
    bins, hist_range : histogram binning passed to np.histogram.
    k : half-width of the refit window in units of the fitted sigma.
    n_iter : number of refit passes.
    init_window_frac : initial window = contiguous bins above this fraction
        of the peak height (i.e. roughly the FWHM region to start from).

    Returns
    -------
    dict with: mode, mode_err, sigma, sigma_err, amp, popt, fit_lo, fit_hi,
    and the histogram (centers, counts) for plotting.
    """
    samples = np.asarray(samples, dtype=float)
    counts, edges = np.histogram(samples, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peak_i = int(counts.argmax())
    thresh = init_window_frac * counts[peak_i]
    lo_i = peak_i
    while lo_i > 0 and counts[lo_i - 1] >= thresh:
        lo_i -= 1
    hi_i = peak_i
    while hi_i < len(counts) - 1 and counts[hi_i + 1] >= thresh:
        hi_i += 1
    lo, hi = centers[lo_i], centers[hi_i]

    popt = None
    pcov = None
    for _ in range(n_iter):
        mask = (centers >= lo) & (centers <= hi)
        x, y = centers[mask], counts[mask].astype(float)
        if x.size < 4:
            break
        yerr = np.sqrt(np.maximum(y, 1.0))          # Poisson, floored at 1
        if popt is None:
            p0 = [y.max(), x[y.argmax()], max((hi - lo) / 4.0, 1e-6)]
        else:
            p0 = popt
        popt, pcov = curve_fit(_gaussian, x, y, p0=p0, sigma=yerr,
                               absolute_sigma=True, maxfev=20000)
        mu, sigma = popt[1], abs(popt[2])
        lo, hi = mu - k * sigma, mu + k * sigma

    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 3
    return {
        "mode": float(popt[1]), "mode_err": float(perr[1]),
        "sigma": float(abs(popt[2])), "sigma_err": float(perr[2]),
        "amp": float(popt[0]), "popt": popt,
        "fit_lo": float(lo), "fit_hi": float(hi),
        "centers": centers, "counts": counts,
    }


def bootstrap_stats(samples, n_boot=500, seed=0, fit_mode=True, bins=400,
                    hist_range=None):
    """Mean, median (and optionally a Gaussian-fit mode) with bootstrap errors.

    Each error is the standard deviation of the statistic across `n_boot`
    resamples of the data. For the mode we re-run the Gaussian peak fit on
    every resample, which is the most honest mode uncertainty (it folds in
    both Poisson noise and fit instability).
    """
    samples = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    n = samples.size

    means = np.empty(n_boot)
    medians = np.empty(n_boot)
    modes = np.full(n_boot, np.nan) if fit_mode else None

    for b in range(n_boot):
        s = samples[rng.integers(0, n, n)]
        means[b] = s.mean()
        medians[b] = np.median(s)
        if fit_mode:
            try:
                modes[b] = gaussian_peak_fit(s, bins=bins,
                                             hist_range=hist_range)["mode"]
            except Exception:
                pass  # leave as NaN; skipped below

    out = {
        "mean": float(samples.mean()),
        "mean_err": float(means.std(ddof=1)),
        "median": float(np.median(samples)),
        "median_err": float(medians.std(ddof=1)),
    }
    if fit_mode:
        m = modes[np.isfinite(modes)]
        # point estimate from the full sample; error from the bootstrap spread
        full = gaussian_peak_fit(samples, bins=bins, hist_range=hist_range)
        out["mode"] = full["mode"]
        out["mode_err"] = float(m.std(ddof=1)) if m.size > 1 else full["mode_err"]
        out["_fit"] = full
    return out


def summarize(samples, name="dE", n_boot=500, bins=400, hist_range=None,
              seed=0):
    """Convenience: compute everything and return a printable dict."""
    res = bootstrap_stats(samples, n_boot=n_boot, seed=seed, bins=bins,
                          hist_range=hist_range)
    print(f"[{name}]  N = {len(samples):,}")
    print(f"  mode   = {res['mode']:.4f} +/- {res['mode_err']:.4f}   (Gaussian peak fit)")
    print(f"  median = {res['median']:.4f} +/- {res['median_err']:.4f}   (bootstrap)")
    print(f"  mean   = {res['mean']:.4f} +/- {res['mean_err']:.4f}   (bootstrap; tail-sensitive)")
    return res
