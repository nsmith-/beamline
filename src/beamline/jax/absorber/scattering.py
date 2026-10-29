"""Multiple Coulomb scattering samplers (PDG 34.3, 34.22 for Highland,
also Fruhwirth and Regler 2001.

Every sampler here is a callable ``(InteractionParams, key) ->
(theta_x, theta_y, y_x, y_y)``: two projected exit angles [rad] in
orthogonal planes containing the direction of flight, and the matching
lateral offsets [mm] accumulated inside the segment. Any can be
adapted into a kick with ``beamline.jax.integrate.stochastic.scattering_kick``.

Highland (PDG 34.3, 34.22):
``highland_scattering_sampler``: two independent projected planes, each
Gaussian with RMS ``theta0``. Within each plane the exit angle and the
in-material lateral offset are correlated (rho = sqrt(3)/2)) and drawn jointly
from two standard normals.

Fruhwirth mixture models:
``mixture_scattering_sampler`` implements the model in

[FL] R. Fruhwirth, M. Liendl, Comput. Phys. Commun. 141 (2001)
[FR] R. Fruhwirth, M. Regler, Nucl. Instrum. Meth. A 456 (2001)

choosing per segment, as [FL] does, among four regimes keyed on the mean number
of single scatters ``N`` ([FL] Eq. 4) and the reduced thickness
``d' = d / (beta^2 X0)``.

    FEW (N < 10):                               Poisson sum of single scatters
                                                ([FL] 235)
    SEMI_GAUSSIAN (d' < 0.6 / Z^0.6):           Gaussian core + single
                                                scattering tail ([FL] 234)
    GAUSSIAN_MIXTURE (d' <= 10):                Two Gaussians ([FL] 233; the
                                                epsilon <= 0 case)
    GAUSSIAN (d' > 10):                         One Gaussian ([FL] 234; CLT
                                                takeover with negligible tails)

Normalization:

The mixture describes nuclear scattering
only, as in the single-scattering model it was fitted to ([FR] Eq. 1, Z^2), with
the total projected width from the nuclear scattering length X_s ([FR] Eq. 40)

    sigma_T = 15 MeV * |z| / (beta p) * sqrt(d / X_s),   X_s = h(Z) X0,

and the Gaussian-mixture fits in terms of d / (beta^2 X_s) ([FR] Eqs. 64-65).
Scattering on atomic electrons is added as an independent Gaussian. Each
electron scatter is limited to theta_e = m_e / m (RF-Track, Stechauner et al.,
IPAC'24 WEPR31, Eqs. 5-6), so with the same screening angle a the electron
variance relative to the nuclear one is

    sigma_e^2 / sigma_T^2 = r_e [ln(theta_e / a) - 0.5] / [ln(b / a) - 0.5],

r_e = sum n_i Z_i / sum n_i Z_i^2 (1/Z for an element).

The published model ([FL] Eq. 5) normalizes to X0 instead, with no separate
electron term. That implicitly gives atomic electrons the nuclear (large-angle)
tail and overestimates the width at low Z: h(Z) = X_s/X0 is 2.2 for hydrogen,
1.5 for LiH and 1.1 for aluminium.

Regime boundaries are as published (in d/(beta^2 X0)).

``screening=`` ("moliere" default, "jackson"): the single-scatter screening
angle. The published fits used Jackson's rough
Thomas-Fermi screening (radius 1.4 a_0 Z^-1/3); Moliere's chi_a is ~1.7 times
larger, giving fewer, larger single scatters (same total variance) and a narrower
core. Against Geant4 (G4beamline 3.08) this takes the LH2 core width from +5% to
+2% and LiH from +8% to +4%. It acts in the FEW and SEMI_GAUSSIAN regimes only;
the GAUSSIAN_MIXTURE fits ([FR] Eqs. 64-65) depend on thickness alone.
[FL] Sec. 4 shows empirically that the convolution property holds.
"""

from __future__ import annotations

import typing
from enum import IntEnum

import hepunits as u
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from beamline.jax.absorber.material import InteractionParams
from beamline.jax.types import SFloat
from beamline.units import ELECTRON_MASS


def highland_scattering_sampler(
    params: InteractionParams, key: Array
) -> tuple[SFloat, SFloat, SFloat, SFloat]:
    """Sample correlated (angle, offset) pairs in two planes.

    For each plane, with two standard normals z1, z2:
        theta = z2 * theta0
            y = z1 * (x*theta0/sqrt(12)) + z2 * (x*theta0/2)
    so angle and offset share z2. ``params`` supplies theta0 and the segment
    thickness x.

    Returns:
        (theta_x, theta_y, y_x, y_y).
    """
    theta0 = params.theta0
    x = params.thickness
    kx, ky = jr.split(key)
    k1x, k2x = jr.split(kx)
    k1y, k2y = jr.split(ky)
    z1x, z2x = jr.normal(k1x), jr.normal(k2x)
    z1y, z2y = jr.normal(k1y), jr.normal(k2y)
    a = x * theta0 / jnp.sqrt(12.0)
    b = x * theta0 / 2.0
    theta_x, theta_y = z2x * theta0, z2y * theta0
    y_x = z1x * a + z2x * b
    y_y = z1y * a + z2y * b
    return theta_x, theta_y, y_x, y_y


# Parameters for the Fruhwirth mixture model

MCS_SCALE: float = 15.0 * u.MeV
"""Scale of the total projected width sigma_T"""
FEW_SCATTER_THRESHOLD: float = 10.0
"""Below this mean number of single scatters, sum them explicitly"""
FEW_SCATTER_MAX: int = 32
"""Fixed number of single-scatter slots in the few-scatter regime

The count is Poisson with mean below ``FEW_SCATTER_THRESHOLD``; the
probability of exceeding 32 slots at mean 10 is about 2e-8, and excess
events are dropped."""
GAUSSIAN_THRESHOLD: float = 10.0
"""Above this d/(beta^2 X0) the tail is negligible"""


SCREENINGS: tuple[str, ...] = ("jackson", "moliere")
DEFAULT_SCREENING: str = "moliere"
FINE_STRUCTURE: float = 1.0 / 137.035999


class Regime(IntEnum):
    """Which mixture model a segment uses (see module docstring)"""

    FEW = 0
    SEMI_GAUSSIAN = 1
    GAUSSIAN_MIXTURE = 2
    GAUSSIAN = 3


class MixtureParams(typing.NamedTuple):
    """Everything the mixture samplers get from ``InteractionParams``

    Widths are in radians (already multiplied by sigma_T) so that
    they can be used directly. The dimensionless standard-measure
    values of the papers are ``width / sigma_total``.
    """

    regime: Array
    """``Regime`` selected for this segment (int)"""
    sigma_total: SFloat
    """Total projected RMS angle sigma_T [rad]"""
    mean_scatters: SFloat
    """Mean number of single scatters N in the segment"""
    reduced_thickness: SFloat
    """d' = d / (beta^2 X0)"""
    few_a: SFloat
    """Few-scatter: single-scatter screening angle [rad]"""
    rho: SFloat
    """Ratio of maximum to minimum single-scatter angle b/a"""
    semi_eps: SFloat
    """Semi-Gaussian: weight of the single-scattering tail"""
    semi_sigma: SFloat
    """Semi-Gaussian: core RMS [rad]"""
    semi_a: SFloat
    """Semi-Gaussian: tail screening angle [rad]"""
    semi_b: SFloat
    """Semi-Gaussian: tail cutoff angle [rad]"""
    gmix_eps: SFloat
    """Gaussian mixture: tail weight"""
    gmix_sigma1: SFloat
    """Gaussian mixture: core RMS [rad]"""
    gmix_sigma2: SFloat
    """Gaussian mixture: tail RMS [rad]"""
    sigma_electron: SFloat
    """Per-plane RMS of the independent atomic-electron Gaussian [rad].
    The total projected variance is
    ``sigma_total**2 + sigma_electron**2``."""


def _scatters_per_radiation_length(Z: SFloat, beta: SFloat, charge: SFloat) -> SFloat:
    """N0, mean number of single scatters per radiation length.
    Cross section scales as charge^2."""

    return (
        1.587e7
        * Z ** (1.0 / 3.0)
        / (beta**2 * (Z + 1.0) * jnp.log(287.0 / jnp.sqrt(Z)))
    )


def _semi_gaussian_shape(N: SFloat, Z: SFloat, rho: SFloat):
    """Standard-measure (eps, sigma^2, a, b) of the semi-Gaussian"""
    n = Z**0.1 * jnp.log(N)
    var = 1.827e-1 + 3.803e-2 * n + 5.783e-4 * n**2
    a = 2.822e-1 + 9.828e-2 * n - 1.355e-2 * n**2 + 1.330e-3 * n**3 - 4.590e-5 * n**4
    b = rho / jnp.sqrt(N * (jnp.log(rho) - 0.5))
    tail_var = a**2 * (jnp.log(b / a) - 0.5)
    eps = (1.0 - var) / (tail_var - var)
    return eps, var, a, b


def _gaussian_mixture_shape_xs(d_red_s: SFloat, Z: SFloat):
    """Standard-measure (eps, sigma1^2, sigma2^2) of the Gaussian mixture with the
    fits in terms of d' = d/(beta^2 X_s) ([FR] Eqs. 64-65)"""
    L1 = jnp.log(d_red_s)
    var1 = 0.8510 + 0.03314 * L1 - 0.001825 * L1**2
    L2 = jnp.log(Z ** (2.0 / 3.0) * d_red_s)
    eps = jnp.where(
        L2 < 0.5,
        0.05283 + 0.0077 * L2 + 0.00069 * L2**2,
        -0.01517 + 0.1151 * L2 - 0.00653 * L2**2,
    )
    var2 = (1.0 - (1.0 - eps) * var1) / eps
    return eps, var1, var2


def mixture_parameters(
    params: InteractionParams,
    screening: str = DEFAULT_SCREENING,
) -> MixtureParams:
    """Derive the regime and all mixture parameters for one segment

    ``screening`` selects the single-scatter screening angle:

    ``"jackson"``: a = alpha Z^1/3 m_e c / (1.4 p), screening radius
    1.4 a_0 Z^-1/3 ([FR] Eqs. 5-6), as used for the published fits.
    ``"moliere"``: Moliere's chi_a = chi_0 sqrt(1.13 + 3.76 (alpha Z/beta)^2),
    chi_0 = hbar/(p 0.885 a_0 Z^-1/3), about 1.7 times larger. The nuclear cutoff
    b and the total nuclear variance are kept, so there are fewer, larger
    scatters (N a^2 (ln(b/a) - 0.5) fixed) and a heavier tail.
    """
    if screening not in SCREENINGS:
        msg = f"screening must be one of {SCREENINGS}, not {screening!r}"
        raise ValueError(msg)
    Z = jnp.asarray(params.Z, dtype=float)
    beta = params.beta
    x0 = params.radiation_lengths
    # h = X_s / X0; with h = 1 the nuclear part reduces to the published model
    h = params.scattering_length_ratio
    scale = MCS_SCALE * jnp.abs(params.charge) / (beta * params.momentum)
    sigma_total = scale * jnp.sqrt(x0 / h)
    N0 = _scatters_per_radiation_length(Z, beta, params.charge)
    N = x0 * N0
    d_red = x0 / beta**2
    rho = 41000.0 / Z ** (2.0 / 3.0)

    # Few scatters: standardized screening angle 1/sqrt(N (ln rho - 0.5)) times
    # sigma_T, so that N single scatters carry exactly sigma_T^2. Written in the
    # d-independent form sigma_T^2 / N = scale^2 / (h N0), which stays finite as
    # d -> 0. With h = X_s/X0 this is the physical screening angle of [FR] Eq. 6.
    few_a = scale / jnp.sqrt(h * N0 * (jnp.log(rho) - 0.5))

    if screening == "moliere":
        # keep b = rho a and N a^2 (ln rho - 0.5) (the nuclear variance) fixed
        f = (1.4 / 0.885) * jnp.sqrt(1.13 + 3.76 * (FINE_STRUCTURE * Z / beta) ** 2)
        rho_m = rho / f
        N0 = N0 * (jnp.log(rho) - 0.5) / (f**2 * (jnp.log(rho_m) - 0.5))
        N = x0 * N0
        few_a = few_a * f
        rho = rho_m

    # Clamp each regime's input to its domain so unselected branches are finite
    crossover = 0.6 / Z**0.6
    N_crossover = crossover * beta**2 * N0
    semi_eps, semi_var, semi_a, semi_b = _semi_gaussian_shape(
        jnp.clip(N, FEW_SCATTER_THRESHOLD, N_crossover), Z, rho
    )
    gmix_eps, gmix_var1, gmix_var2 = _gaussian_mixture_shape_xs(
        jnp.clip(d_red, crossover, GAUSSIAN_THRESHOLD) / h, Z
    )

    # Atomic electrons: same screening angle a, cut off at theta_e = m_e/m
    mass = params.momentum * jnp.sqrt(1.0 - beta**2) / beta  # m c^2 [MeV]
    theta_e = ELECTRON_MASS * u.c_light_sq / mass
    log_e = jnp.maximum(jnp.log(theta_e / few_a) - 0.5, 0.0)
    sigma_electron = sigma_total * jnp.sqrt(
        params.electron_ratio * log_e / (jnp.log(rho) - 0.5)
    )

    regime = jnp.where(
        N < FEW_SCATTER_THRESHOLD,
        Regime.FEW,
        jnp.where(
            (d_red < crossover) & (semi_eps > 0.0),
            Regime.SEMI_GAUSSIAN,
            jnp.where(
                d_red <= GAUSSIAN_THRESHOLD, Regime.GAUSSIAN_MIXTURE, Regime.GAUSSIAN
            ),
        ),
    )
    return MixtureParams(
        regime=regime,
        sigma_total=sigma_total,
        mean_scatters=N,
        reduced_thickness=d_red,
        few_a=few_a,
        rho=rho,
        semi_eps=jnp.clip(semi_eps, 0.0, 1.0),
        semi_sigma=sigma_total * jnp.sqrt(semi_var),
        semi_a=sigma_total * semi_a,
        semi_b=sigma_total * semi_b,
        gmix_eps=gmix_eps,
        gmix_sigma1=sigma_total * jnp.sqrt(gmix_var1),
        gmix_sigma2=sigma_total * jnp.sqrt(gmix_var2),
        sigma_electron=sigma_electron,
    )


# Fruhwirth mixture models: draws

type _Draw = tuple[SFloat, SFloat, SFloat, SFloat]


def _gaussian_draw(key: Array, sigma: SFloat, x: SFloat) -> _Draw:
    """2D Gaussian angle of per-plane RMS ``sigma``, equivalent to
    theta = sigma sqrt(-2 ln u) with uniform azimuth ([FL] 20).
    """
    z = jr.normal(key, (4,))
    tx, ty = sigma * z[0], sigma * z[1]
    yx = x * (tx / 2.0 + sigma * z[2] / jnp.sqrt(12.0))
    yy = x * (ty / 2.0 + sigma * z[3] / jnp.sqrt(12.0))
    return tx, ty, yx, yy


def _single_scatter_angle(u_: Array, a: SFloat, rho: SFloat) -> Array:
    """Inverse CDF of the space single-scattering density ([FL] Eqs. 1, 16)

    theta = a b sqrt((1-u)/(u b^2 + a^2)) with b = rho a rearranged.
    """
    return a * rho * jnp.sqrt((1.0 - u_) / (u_ * rho**2 + 1.0))


def _single_scatter_draw(key: Array, a: SFloat, rho: SFloat, x: SFloat) -> _Draw:
    """One hard single scatter at a uniform depth inside the segment"""
    ku, kphi, kt = jr.split(key, 3)
    theta = _single_scatter_angle(jr.uniform(ku), a, rho)
    phi = jr.uniform(kphi, maxval=2.0 * jnp.pi)
    tx, ty = theta * jnp.cos(phi), theta * jnp.sin(phi)
    lever = x * (1.0 - jr.uniform(kt))
    return tx, ty, lever * tx, lever * ty


def _few_scatter_draw(
    key: Array, mean: SFloat, a: SFloat, rho: SFloat, x: SFloat
) -> _Draw:
    """Poisson-distributed number of single scatters, summed in projection

    The events are generated as a Poisson process along the segment: depth
    fractions are cumulative sums of Exp(1) spacings divided by ``mean``, and
    the events with depth < 1 are kept.
    """
    ke, ku, kphi = jr.split(key, 3)
    depth = jnp.cumsum(jr.exponential(ke, (FEW_SCATTER_MAX,))) / jnp.maximum(
        mean, 1e-300
    )
    active = depth < 1.0
    theta = _single_scatter_angle(jr.uniform(ku, (FEW_SCATTER_MAX,)), a, rho)
    phi = jr.uniform(kphi, (FEW_SCATTER_MAX,), maxval=2.0 * jnp.pi)
    ux = jnp.where(active, theta * jnp.cos(phi), 0.0)
    uy = jnp.where(active, theta * jnp.sin(phi), 0.0)
    lever = x * jnp.where(active, 1.0 - depth, 0.0)
    return ux.sum(), uy.sum(), (lever * ux).sum(), (lever * uy).sum()


def _select(cond: Array, a: _Draw, b: _Draw) -> _Draw:
    return tuple(jnp.where(cond, ai, bi) for ai, bi in zip(a, b, strict=True))  # type: ignore[return-value]


def _add_electrons(draw: _Draw, key: Array, mp: MixtureParams, x: SFloat) -> _Draw:
    """Add the independent atomic-electron Gaussian (angles and offsets)"""
    # fold_in: the electron draw does not change the nuclear draws
    e = _gaussian_draw(jr.fold_in(key, 0x5E), mp.sigma_electron, x)
    return tuple(d + ei for d, ei in zip(draw, e, strict=True))  # type: ignore[return-value]


def mixture_scattering_sampler(
    params: InteractionParams,
    key: Array,
    screening: str = DEFAULT_SCREENING,
) -> tuple[SFloat, SFloat, SFloat, SFloat]:
    """Fruhwirth mixture-model scattering with automatic regime selection

    See the module docstring for the regimes. Drop-in replacement for
    ``highland_scattering_sampler``.

    Returns:
        (theta_x, theta_y, y_x, y_y).
    """
    mp = mixture_parameters(params, screening)
    x = params.thickness
    k_pick, k_core, k_tail, k_semi_tail, k_few, k_gauss = jr.split(key, 6)
    pick = jr.uniform(k_pick)

    few = _few_scatter_draw(k_few, mp.mean_scatters, mp.few_a, mp.rho, x)
    semi = _select(
        pick < mp.semi_eps,
        _single_scatter_draw(k_semi_tail, mp.semi_a, mp.semi_b / mp.semi_a, x),
        _gaussian_draw(k_core, mp.semi_sigma, x),
    )
    gmix = _select(
        pick < mp.gmix_eps,
        _gaussian_draw(k_tail, mp.gmix_sigma2, x),
        _gaussian_draw(k_core, mp.gmix_sigma1, x),
    )
    gauss = _gaussian_draw(k_gauss, mp.sigma_total, x)

    out = gauss
    out = _select(mp.regime == Regime.GAUSSIAN_MIXTURE, gmix, out)
    out = _select(mp.regime == Regime.SEMI_GAUSSIAN, semi, out)
    out = _select(mp.regime == Regime.FEW, few, out)
    return _add_electrons(out, key, mp, x)
