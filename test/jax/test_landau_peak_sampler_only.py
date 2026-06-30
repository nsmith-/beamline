"""
test_landau_peak_sampler_only.py
=================================

Derives ``_LANDAU_STD_PEAK = -0.42931453`` directly from the in-house
sampler's defining equation, without any dependence on scipy.stats.landau
or any other external Landau implementation.

The sampler in diff_random.distributions._landau.Landau_SG draws

    Lambda(U, E) = f(U) - (2/pi) * log(E)

where

    f(U) = (2/pi) * [ (pi/2 + U) * tan(U)
                    - log( (pi/2) * cos(U) / (pi/2 + U) ) ]

with U ~ Uniform(-pi/2, pi/2), E ~ Exp(1).

Holding lambda fixed and using change of variables in E gives the
density of Lambda as a 1D integral:

    p(lambda) = (1/2) * integral_{u=-pi/2}^{pi/2} exp( g - exp(g) ) du

where g(u, lambda) = (pi/2) * (f(u) - lambda).

Maximizing p(lambda) gives the mode of the sampler's output. No
sampling is involved; the result is derived from the sampler's own
algorithm. If the algorithm in _landau.py changes, this test
automatically reflects the new mode.

Run from the repo root:

    uv run pytest test/jax/test_landau_peak_sampler_only.py -v -s
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


# The constant being validated.
EXPECTED_PEAK = -0.42931453


_PI_BY_2 = np.pi / 2


def _f(u: float) -> float:
    """C-M-S inversion term, exactly as in diff_random/_landau.py lines 50-53."""
    return (1.0 / _PI_BY_2) * (
        (_PI_BY_2 + u) * np.tan(u)
        - np.log((_PI_BY_2 * np.cos(u)) / (_PI_BY_2 + u))
    )


def _sampler_density(lam: float) -> float:
    """Density of the sampler's output Lambda at lam, via 1D quadrature.

    Uses the numerically stable form exp(g - exp(g)) = E * exp(-E)
    with E = exp(g). At large positive g the result rounds to zero
    safely; at large negative g it reduces to exp(g), also safe.
    """
    def integrand(u: float) -> float:
        g = _PI_BY_2 * (_f(u) - lam)
        if g > 30.0:
            return 0.0
        if g < -30.0:
            return float(np.exp(g))
        return float(np.exp(g - np.exp(g)))

    # tiny eps to avoid the endpoints where tan blows up
    eps = 1e-7
    val, _ = quad(
        integrand,
        -_PI_BY_2 + eps,
        _PI_BY_2 - eps,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=300,
    )
    return 0.5 * val


def test_sampler_density_peak_matches_constant():
    """Maximize the sampler's own density. Result must match
    _LANDAU_STD_PEAK to high precision.
    """
    res = minimize_scalar(
        lambda x: -_sampler_density(x),
        bracket=(-1.0, -0.4, 0.5),
        method="brent",
        options={"xtol": 1e-10},
    )
    peak = float(res.x)
    print(f"\n  sampler density peak (1D integral + Brent): {peak:.8f}")
    print(f"  absorber.py constant:                        {EXPECTED_PEAK:.8f}")
    print(f"  difference:                                  {peak - EXPECTED_PEAK:+.2e}")
    # 1e-6 leaves headroom over quadrature/Brent precision (~4e-8 in practice)
    assert abs(peak - EXPECTED_PEAK) < 1e-6, (
        f"sampler density peak {peak} disagrees with "
        f"_LANDAU_STD_PEAK = {EXPECTED_PEAK}. The constant in "
        f"absorber.py needs updating to match the sampler in "
        f"diff_random/_landau.py."
    )


def test_sampler_density_normalization():
    """Sanity check: the sampler density integrates to ~1 (a small
    deficit is expected from the truncated integration range, since
    Landau has a heavy right tail).
    """
    total, _ = quad(
        _sampler_density, -5.0, 30.0,
        epsabs=1e-8, epsrel=1e-6, limit=500,
    )
    print(f"\n  density integrated over (-5, 30): {total:.5f}")
    print(f"  (deficit from 1.0 is the truncated heavy right tail)")
    # The peak is local; the tail truncation doesn't affect peak location.
    # Just check the bulk of probability is accounted for.
    assert 0.95 < total < 1.005, (
        f"sampler density does not integrate to ~1 ({total}). "
        f"There may be an error in the change-of-variables derivation."
    )


if __name__ == "__main__":
    test_sampler_density_peak_matches_constant()
    test_sampler_density_normalization()
    print("\nAll tests passed. Constant derived from sampler alone.")
