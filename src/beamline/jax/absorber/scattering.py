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

All four share the total projected standard deviation ([FL] Eq. 5)

    sigma_T = 15 MeV * |z| / (beta p) * sqrt(d / X0),

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


def _gaussian_mixture_shape(d_red: SFloat, Z: SFloat):
    """Standard-measure (eps, sigma1^2, sigma2^2) of the Gaussian mixture"""
    L1 = jnp.log(d_red)
    var1 = 8.471e-1 + 3.347e-2 * L1 - 1.843e-3 * L1**2
    L2 = jnp.log(Z ** (2.0 / 3.0) * d_red)
    eps = jnp.where(
        L2 < 0.5,
        4.841e-2 + 6.348e-3 * L2 + 6.096e-4 * L2**2,
        -1.908e-2 + 1.106e-1 * L2 - 5.729e-3 * L2**2,
    )
    var2 = (1.0 - (1.0 - eps) * var1) / eps
    return eps, var1, var2


def mixture_parameters(params: InteractionParams) -> MixtureParams:
    """Derive the regime and all mixture parameters for one segment"""
    Z = jnp.asarray(params.Z, dtype=float)
    beta = params.beta
    x0 = params.radiation_lengths
    sigma_total = (
        MCS_SCALE * jnp.abs(params.charge) / (beta * params.momentum) * jnp.sqrt(x0)
    )
    N0 = _scatters_per_radiation_length(Z, beta, params.charge)
    N = x0 * N0
    d_red = x0 / beta**2
    rho = 41000.0 / Z ** (2.0 / 3.0)

    # Few scatters: standardized screening angle 1/sqrt(N (ln rho - 0.5)) times
    # sigma_T, so that N single scatters carry exactly sigma_T^2. Written in the
    # d-independent form sigma_T^2 / N = (15 MeV z/(beta p))^2 / N0, which stays
    # finite as d -> 0.
    few_a = (MCS_SCALE * jnp.abs(params.charge) / (beta * params.momentum)) / jnp.sqrt(
        N0 * (jnp.log(rho) - 0.5)
    )

    # Clamp each regime's input to its domain so unselected branches are finite
    crossover = 0.6 / Z**0.6
    N_crossover = crossover * beta**2 * N0
    semi_eps, semi_var, semi_a, semi_b = _semi_gaussian_shape(
        jnp.clip(N, FEW_SCATTER_THRESHOLD, N_crossover), Z, rho
    )
    gmix_eps, gmix_var1, gmix_var2 = _gaussian_mixture_shape(
        jnp.clip(d_red, crossover, GAUSSIAN_THRESHOLD), Z
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


def mixture_scattering_sampler(
    params: InteractionParams, key: Array
) -> tuple[SFloat, SFloat, SFloat, SFloat]:
    """Fruhwirth mixture-model scattering with automatic regime selection

    See the module docstring for the regimes. Drop-in replacement for
    ``highland_scattering_sampler``.

    Returns:
        (theta_x, theta_y, y_x, y_y).
    """
    mp = mixture_parameters(params)
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
    return _select(mp.regime == Regime.FEW, few, out)


def gaussian_mixture_scattering_sampler(
    params: InteractionParams, key: Array
) -> tuple[SFloat, SFloat, SFloat, SFloat]:
    """Two-component Gaussian mixture only ([FL] 2.3), at any thickness

    The purely Gaussian mixture is the one usable in a Gaussian-sum filter and
    is smooth in the beam parameters apart from the component choice. Below the
    semi-Gaussian crossover it underestimates the far tails ([FL] Figs. 3-4);
    the parameterisation is evaluated at ``max(d', 1e-6)`` so very thin
    segments remain well defined. This is for reconstruction/comparisons.

    Warning: below the crossover this model does not have the convolution
    property. Both components have finite variance, so summing many thin
    segments drives the result to a Gaussian and the tails disappear (slicing a
    35 cm LH2 absorber into 100 segments removes nearly all events beyond 3
    sigma_T). Use ``mixture_scattering_sampler`` with ``stochastic_solve``
    unless segments are thicker than ``0.6 / Z^0.6`` radiation lengths.
    """
    mp = mixture_parameters(params)
    Z = jnp.asarray(params.Z, dtype=float)
    eps, var1, var2 = _gaussian_mixture_shape(
        jnp.clip(mp.reduced_thickness, 1e-6, GAUSSIAN_THRESHOLD), Z
    )
    k_pick, k_core, k_tail = jr.split(key, 3)
    s = mp.sigma_total
    return _select(
        jr.uniform(k_pick) < eps,
        _gaussian_draw(k_tail, s * jnp.sqrt(var2), params.thickness),
        _gaussian_draw(k_core, s * jnp.sqrt(var1), params.thickness),
    )
