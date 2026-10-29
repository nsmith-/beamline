"""Tests for the multiple-scattering samplers.

Covers the Fruhwirth mixture model (``mixture_scattering_sampler``) against

  * its own parameterisation  -- regime selection and the exact variance
                                 identity sigma_T^2 in every regime
  * the physics it models     -- a brute-force Poisson sum of single scatters
                                 (the reference used in Fruhwirth & Regler,
                                 NIM A 456 (2001) 369, Fig. 3)
  * the integrator's needs    -- invariance under slicing (convolution
                                 property), finite gradients at zero
                                 deflection, end-to-end in ``stochastic_solve``

Heavy tails make the sample variance a poor statistic for thin segments (it is
dominated by events too rare to be drawn), so shape comparisons use tail
fractions P(|theta_x| > k sigma_T) and a MAD-based width instead.
"""

import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.scattering import (
    Regime,
    highland_scattering_sampler,
    mixture_parameters,
    mixture_scattering_sampler,
)
from beamline.jax.absorber.straggling import landau_energy_loss_sampler
from beamline.jax.absorber.volume import (
    AbsorberCylinder,
    SumMaterialVolume,
    TransformMaterialVolume,
)
from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrate.stochastic import (
    StochasticKick,
    apply_scattering,
    energy_loss_kick,
    scattering_kick,
    stochastic_solve,
)
from beamline.jax.kinematics import MuonStateDz

SEED = 7
TAIL_KS = (1.0, 2.0, 3.0, 5.0, 10.0)


def make_params(material: str, thickness: float, pz: float = 200.0 * u.MeV):
    muon = MuonStateDz.make(
        position=Cartesian4.make(), momentum=Cartesian3.make(z=pz), q=1
    )
    return MATERIALS[material].interaction_params(muon, thickness)


def draw(sampler, params, n: int, seed: int = SEED) -> np.ndarray:
    """(n, 4) array of (theta_x, theta_y, y_x, y_y)"""
    out = jax.jit(jax.vmap(lambda k: jnp.stack(sampler(params, k))))(
        jr.split(jr.key(seed), n)
    )
    return np.asarray(out)


def single_scatter_sum(params, n: int, seed: int = SEED, chunk: int = 20_000):
    """Reference: projected angle from a Poisson number of single scatters

    Uses the same single-scattering density (screening angle and cutoff) that
    the mixture models were fitted to, normalised so N scatters carry sigma_T^2,
    plus the independent atomic-electron Gaussian.
    """
    mp = mixture_parameters(params)
    sigma_e = float(mp.sigma_electron)
    mean, a, rho = float(mp.mean_scatters), float(mp.few_a), float(mp.rho)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n // chunk):
        counts = rng.poisson(mean, chunk)
        total = int(counts.sum())
        uu = rng.random(total)
        theta = a * rho * np.sqrt((1.0 - uu) / (uu * rho**2 + 1.0))
        phi = rng.random(total) * 2.0 * np.pi
        owner = np.repeat(np.arange(chunk), counts)
        nuc = np.bincount(owner, theta * np.cos(phi), minlength=chunk)
        out.append(nuc + rng.normal(0.0, sigma_e, chunk))
    return np.concatenate(out)


def tail_fractions(theta_over_sigma: np.ndarray) -> np.ndarray:
    a = np.abs(theta_over_sigma)
    return np.array([np.mean(a > k) for k in TAIL_KS])


def robust_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


# --- parameterization --------------------------------------------------------


@pytest.mark.parametrize(
    ("thickness", "expected"),
    [
        (1e-4 * u.mm, Regime.FEW),
        (5.0 * u.mm, Regime.SEMI_GAUSSIAN),
        (349.6 * u.mm, Regime.SEMI_GAUSSIAN),
        (5.0 * u.m, Regime.GAUSSIAN_MIXTURE),
        (100.0 * u.m, Regime.GAUSSIAN),
    ],
)
def test_regime_selection(thickness, expected):
    """Regimes follow N < 10, d' < 0.6/Z^0.6, d' <= 10"""
    mp = mixture_parameters(make_params("liquid_hydrogen_H2", thickness))
    assert Regime(int(mp.regime)) == expected


@pytest.mark.parametrize(
    "material", ["liquid_hydrogen_H2", "lithium_hydride_MICE", "silicon_dioxide_SiO2"]
)
@pytest.mark.parametrize("thickness", [1e-4, 1e-2, 1.0, 5.0, 100.0, 5e3, 2e5])
def test_variance_identity(material, thickness):
    """Every regime's analytic variance equals sigma_T^2 (additive over slices)"""
    mp = mixture_parameters(make_params(material, thickness * u.mm))
    sT2 = float(mp.sigma_total) ** 2
    regime = Regime(int(mp.regime))
    if regime == Regime.FEW:
        var = float(mp.mean_scatters * mp.few_a**2 * (jnp.log(mp.rho) - 0.5))
    elif regime == Regime.SEMI_GAUSSIAN:
        tail = mp.semi_a**2 * (jnp.log(mp.semi_b / mp.semi_a) - 0.5)
        var = float((1 - mp.semi_eps) * mp.semi_sigma**2 + mp.semi_eps * tail)
    elif regime == Regime.GAUSSIAN_MIXTURE:
        var = float(
            (1 - mp.gmix_eps) * mp.gmix_sigma1**2 + mp.gmix_eps * mp.gmix_sigma2**2
        )
    else:
        var = sT2
    assert var == pytest.approx(sT2, rel=1e-9)
    assert 0.0 <= float(mp.semi_eps) <= 1.0
    assert 0.0 < float(mp.gmix_eps) < 0.5


@pytest.mark.parametrize("thickness", [5.0 * u.m, 100.0 * u.m])
def test_sample_variance_thick(thickness):
    """Where the tails are Gaussian, the sample variance is sigma_T^2 + sigma_e^2"""
    params = make_params("silicon_dioxide_SiO2", thickness)
    mp = mixture_parameters(params)
    sT = float(np.hypot(mp.sigma_total, mp.sigma_electron))
    s = draw(mixture_scattering_sampler, params, 200_000)
    assert np.var(s[:, 0]) / sT**2 == pytest.approx(1.0, rel=0.02)
    assert np.var(s[:, 1]) / sT**2 == pytest.approx(1.0, rel=0.02)


# --- physics -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("material", "thickness"),
    [("liquid_hydrogen_H2", 1.0 * u.mm), ("lithium_hydride_MICE", 0.1 * u.mm)],
)
def test_matches_single_scatter_sum(material, thickness):
    """Semi-Gaussian tails agree with a brute-force single-scatter sum

    These Z = 1, 2 cases are an extrapolation of fits made for C, Si and Fe.
    Highland, by contrast, has essentially no events beyond 3 sigma_T.
    """
    params = make_params(material, thickness)
    mp = mixture_parameters(params)
    assert Regime(int(mp.regime)) == Regime.SEMI_GAUSSIAN
    sT = float(np.hypot(mp.sigma_total, mp.sigma_electron))  # total width
    n = 200_000
    ref = tail_fractions(single_scatter_sum(params, n) / sT)
    mix = tail_fractions(draw(mixture_scattering_sampler, params, n)[:, 0] / sT)
    hl = tail_fractions(draw(highland_scattering_sampler, params, n)[:, 0] / sT)
    ratio = mix / ref
    assert np.all((ratio > 0.7) & (ratio < 1.4)), dict(zip(TAIL_KS, ratio, strict=True))
    assert hl[TAIL_KS.index(3.0)] < 0.05 * ref[TAIL_KS.index(3.0)]


def test_convolution_property():
    """Slicing an absorber leaves the auto-selected mixture distribution unchanged

    35 cm LH2 at 164.9 MeV/c (benchmark 3.1 step scan): one slab vs 100 slices.
    """
    n_events, n_slices, thickness = 100_000, 100, 349.6 * u.mm
    full = make_params("liquid_hydrogen_H2", thickness, 164.9 * u.MeV)
    part = make_params("liquid_hydrogen_H2", thickness / n_slices, 164.9 * u.MeV)
    sT = float(mixture_parameters(full).sigma_total)

    one = draw(mixture_scattering_sampler, full, n_events)[:, 0] / sT

    @jax.jit
    @jax.vmap
    def sliced(key):
        keys = jr.split(key, n_slices)
        return jax.vmap(lambda k: mixture_scattering_sampler(part, k)[0])(keys).sum()

    many = np.asarray(sliced(jr.split(jr.key(SEED + 1), n_events))) / sT
    assert robust_sigma(many) == pytest.approx(robust_sigma(one), rel=0.02)
    ratio = tail_fractions(many)[1:4] / tail_fractions(one)[1:4]  # k = 2, 3, 5
    assert np.all((ratio > 0.75) & (ratio < 1.35)), ratio


def test_planes_uncorrelated_but_dependent():
    """theta_x and theta_y share the mixture component (Fruhwirth & Liendl Sec. 3)

    Uncorrelated by azimuthal symmetry, but large |theta_x| makes large
    |theta_y| more likely, which independent per-plane draws would miss.
    """
    params = make_params("liquid_hydrogen_H2", 5.0 * u.mm)
    s = draw(mixture_scattering_sampler, params, 200_000)
    tx, ty = s[:, 0], s[:, 1]
    assert abs(np.corrcoef(tx, ty)[0, 1]) < 0.01
    big = np.abs(tx) > 3 * robust_sigma(tx)
    assert np.mean(np.abs(ty[big])) > 2.0 * np.mean(np.abs(ty[~big]))


def test_offset_angle_correlation():
    """Gaussian regime reproduces the PDG 34.22 rho = sqrt(3)/2"""
    params = make_params("silicon_dioxide_SiO2", 100.0 * u.m)
    s = draw(mixture_scattering_sampler, params, 200_000)
    assert np.corrcoef(s[:, 0], s[:, 2])[0, 1] == pytest.approx(
        np.sqrt(3) / 2, abs=0.01
    )


# --- integration with the stochastic driver ----------------------------------


def test_zero_deflection_gradient_is_finite():
    """apply_scattering has a finite gradient when the kick is exactly zero"""
    muon = MuonStateDz.make(
        position=Cartesian4.make(), momentum=Cartesian3.make(z=200.0 * u.MeV), q=1
    )

    def f(scale):
        return apply_scattering(muon, 0.0 * scale, 0.0 * scale, 0.0, 0.0).kin.t.x

    assert np.isfinite(float(jax.grad(f)(1.0)))


def test_sampler_gradient_every_regime():
    """d theta / d p is finite, and negative on average, in every regime"""
    for thickness in (1e-4 * u.mm, 5.0 * u.mm, 5.0 * u.m, 100.0 * u.m):

        def msq(pz, thickness=thickness):
            params = make_params("liquid_hydrogen_H2", thickness, pz)
            keys = jr.split(jr.key(SEED), 4096)
            tx = jax.vmap(lambda k: mixture_scattering_sampler(params, k)[0])(keys)
            return jnp.mean(tx**2)

        g = float(jax.grad(msq)(200.0 * u.MeV))
        assert np.isfinite(g)
        assert g <= 0.0


def _run_absorber(sampler, n: int):
    absorber = AbsorberCylinder(
        material=MATERIALS["silicon_dioxide_SiO2"],
        radius=100.0 * u.mm,
        length=10.0 * u.mm,
        char_length=2.0 * u.mm,
    )
    start = MuonStateDz.make(
        position=Cartesian4.make(z=-20.0 * u.mm),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    kick = StochasticKick(
        straggling=energy_loss_kick(landau_energy_loss_sampler),
        scattering=scattering_kick(sampler),
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    zs = jnp.array([-20.0 * u.mm, 20.0 * u.mm])
    run = jax.jit(
        jax.vmap(
            lambda k: stochastic_solve(field, absorber, start, zs, k, kick=kick)[0]
        )
    )
    ys = run(jr.split(jr.key(SEED), n))
    theta_x = np.arctan2(np.asarray(ys.kin.t.x[:, -1]), np.asarray(ys.kin.t.z[:, -1]))
    params = absorber.interaction_params(start, absorber.length)
    return theta_x, float(mixture_parameters(params).sigma_total)


def test_stochastic_solve_end_to_end():
    """Through the integrator the mixture keeps its tails; Highland does not"""
    n = 10_000
    th_mix, sT = _run_absorber(mixture_scattering_sampler, n)
    th_hl, _ = _run_absorber(highland_scattering_sampler, n)
    assert np.all(np.isfinite(th_mix))
    tail_mix = np.mean(np.abs(th_mix) > 3 * sT)
    tail_hl = np.mean(np.abs(th_hl) > 3 * sT)
    assert tail_mix > 3e-3
    assert tail_mix > 3 * tail_hl


@pytest.mark.xfail(
    strict=False,
    reason="OPEN ISSUE: the sign of "
    "this 256-muon gradient estimate through stochastic_solve depends on "
    "the seed (e.g. seeds 7 and 10 give > 0), although single-segment "
    "pathwise gradients match the exact ones (see "
    "test_single_segment_pathwise_gradient).",
)
def test_stochastic_solve_gradient():
    """Reverse-mode gradient through stochastic_solve with the mixture kick"""
    absorber = AbsorberCylinder(
        material=MATERIALS["liquid_hydrogen_H2"],
        radius=100.0 * u.mm,
        length=20.0 * u.mm,
        char_length=5.0 * u.mm,
    )
    kick = StochasticKick(
        straggling=energy_loss_kick(landau_energy_loss_sampler),
        scattering=scattering_kick(mixture_scattering_sampler),
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    zs = jnp.array([-30.0 * u.mm, 30.0 * u.mm])

    def loss(pz):
        start = MuonStateDz.make(
            position=Cartesian4.make(z=-30.0 * u.mm),
            momentum=Cartesian3.make(z=pz),
            q=1,
        )

        def one(k):
            ys, _ = stochastic_solve(field, absorber, start, zs, k, kick=kick)
            return (ys.kin.t.x[-1] / ys.kin.t.z[-1]) ** 2

        return jnp.mean(jax.vmap(one)(jr.split(jr.key(SEED), 256)))

    g = float(jax.jit(jax.grad(loss))(200.0 * u.MeV))
    assert np.isfinite(g)
    assert g < 0.0  # more momentum, less scattering


def _two_absorber_run(material_volume, n: int = 256):
    kick = StochasticKick(
        straggling=energy_loss_kick(landau_energy_loss_sampler),
        scattering=scattering_kick(mixture_scattering_sampler),
    )
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    start = MuonStateDz.make(
        position=Cartesian4.make(z=-10.0 * u.mm),
        momentum=Cartesian3.make(z=200.0 * u.MeV),
        q=1,
    )
    zs = jnp.array([-10.0 * u.mm, 20.0 * u.mm])
    run = jax.jit(
        jax.vmap(
            lambda k: stochastic_solve(field, material_volume, start, zs, k, kick=kick)[
                0
            ]
        )
    )
    ys = run(jr.split(jr.key(SEED), n))
    return np.asarray(ys.kin.t.ct[:, 0] - ys.kin.t.ct[:, -1])


def test_sum_material_volume_exit_segment():
    """Exit segments of SumMaterialVolume components get material params

    Regression: params were evaluated at the step end, which lies outside every
    component on an exiting step, so SumMaterialVolume returned all-zero params
    (no straggling/scattering there; NaN for the mixture sampler).
    """

    def cylinder(zc):
        return TransformMaterialVolume(
            transform=Transform.make_translation(z=zc),
            material=AbsorberCylinder(
                material=MATERIALS["lithium_hydride_LiH"],
                radius=100.0 * u.mm,
                length=5.0 * u.mm,
            ),
        )

    single = cylinder(2.5 * u.mm)
    summed = SumMaterialVolume(components=[single, cylinder(500.0 * u.mm)])
    de_single = _two_absorber_run(single)
    de_sum = _two_absorber_run(summed)
    assert np.all(np.isfinite(de_sum))
    assert np.mean(de_sum) == pytest.approx(np.mean(de_single), rel=1e-6)


# --- artifact ----------------------------------------------------------------


def test_summary_figure(artifacts_dir):
    """Projected-angle densities: models vs single-scatter sum, and slicing"""
    n = 200_000
    edges = np.linspace(-15, 15, 241)
    mid = 0.5 * (edges[1:] + edges[:-1])

    def hist(a):
        return np.histogram(a, bins=edges, density=True)[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

    params = make_params("liquid_hydrogen_H2", 1.0 * u.mm)
    sT = float(mixture_parameters(params).sigma_total)
    axA.step(
        mid,
        hist(single_scatter_sum(params, n) / sT),
        "k",
        where="mid",
        lw=2,
        label="single-scatter sum (reference)",
    )
    for sampler, c in [
        (mixture_scattering_sampler, "C0"),
        (highland_scattering_sampler, "C3"),
    ]:
        axA.step(
            mid,
            hist(draw(sampler, params, n)[:, 0] / sT),
            c,
            where="mid",
            label=sampler.__name__,
        )
    axA.set(
        yscale="log",
        ylim=(1e-6, 2),
        xlabel=r"$\theta_x / \sigma_T$",
        ylabel="density",
        title="1 mm LH$_2$, 200 MeV/c (N $\\approx$ 200 scatters)",
    )
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    thickness = 349.6 * u.mm
    full = make_params("liquid_hydrogen_H2", thickness, 164.9 * u.MeV)
    sT = float(mixture_parameters(full).sigma_total)
    for sampler, ls in [
        (mixture_scattering_sampler, "-"),
    ]:
        for n_slices, c in [(1, "C0"), (10, "C1"), (100, "C2")]:
            part = make_params(
                "liquid_hydrogen_H2", thickness / n_slices, 164.9 * u.MeV
            )

            @jax.jit
            @jax.vmap
            def sliced(key, part=part, n_slices=n_slices, sampler=sampler):
                keys = jr.split(key, n_slices)
                return jax.vmap(lambda k: sampler(part, k)[0])(keys).sum()

            th = np.asarray(sliced(jr.split(jr.key(SEED), n // 2))) / sT
            name = sampler.__name__.replace("_scattering_sampler", "")
            axB.step(
                mid,
                hist(th),
                c,
                ls=ls,
                where="mid",
                label=f"{name}, {n_slices} slice(s)",
            )
    axB.set(
        yscale="log",
        ylim=(1e-6, 2),
        xlabel=r"$\theta_x / \sigma_T$",
        ylabel="density",
        title="Convolution property: 349.6 mm LH$_2$, 164.9 MeV/c",
    )
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(artifacts_dir / "mixture_scattering.png", dpi=130)
    plt.close(fig)


# --- nuclear (X_s) + atomic electrons vs published (X0) normalization ---------


@pytest.mark.parametrize(
    ("material", "expected"),
    [
        ("liquid_hydrogen_H2", 2.23),  # [FR] Eq. 37: h(1) = 2 ln 287 / ln 159
        ("aluminum_Al", 1.12),
        ("silicon_Si", 1.11),
    ],
)
def test_scattering_length_ratio(material, expected):
    """h = X_s/X0 agrees with [FR] Eq. 37 (X0 from the PDG table, so ~1-2%)"""
    assert MATERIALS[material].scattering_length_ratio == pytest.approx(
        expected, rel=0.02
    )


def _sigma_x0(params):
    """Published [FL] Eq. 5 width, 15 MeV / (beta p) sqrt(d/X0)"""
    return float(
        15.0
        * u.MeV
        / (params.beta * params.momentum)
        * np.sqrt(params.radiation_lengths)
    )


def test_electron_ratio():
    """1/Z for elements; sum n Z / sum n Z^2 = 4/10 for LiH"""
    assert MATERIALS["liquid_hydrogen_H2"].electron_ratio == pytest.approx(1.0)
    assert MATERIALS["aluminum_Al"].electron_ratio == pytest.approx(1 / 13)
    for lih in ("lithium_hydride_LiH", "lithium_hydride_MICE"):
        assert MATERIALS[lih].electron_ratio == pytest.approx(0.4, rel=0.01)


@pytest.mark.parametrize("material", ["liquid_hydrogen_H2", "aluminum_Al"])
def test_nuclear_normalization(material):
    """Nuclear sigma_T uses X_s = h X0; the few-scatter screening angle is then
    the physical one of [FR] Eq. 6, a = 2.66e-6 Z^1/3 / p[GeV] (up to the
    41000/Z^2/3 approximation for b/a)"""
    params = make_params(material, 10.0 * u.mm)
    new = mixture_parameters(params, screening="jackson")
    h = float(params.scattering_length_ratio)
    assert float(new.sigma_total) == pytest.approx(_sigma_x0(params) / np.sqrt(h))
    Z = float(params.Z)
    a_phys = 2.66e-6 * Z ** (1 / 3) / (float(params.momentum) / u.GeV)
    assert float(new.few_a) == pytest.approx(a_phys, rel=0.05)


def test_electron_term_size():
    """Electrons add ~half the nuclear variance in H2, a few percent in Al, and the
    total stays below the published X0 width"""
    for material, lo, hi in (
        ("liquid_hydrogen_H2", 0.4, 0.6),
        ("aluminum_Al", 0.02, 0.06),
    ):
        params = make_params(material, 10.0 * u.mm)
        mp = mixture_parameters(params)
        ratio = float(mp.sigma_electron) ** 2 / float(mp.sigma_total) ** 2
        assert lo < ratio < hi, (material, ratio)
        total = float(np.hypot(mp.sigma_total, mp.sigma_electron))
        assert total < _sigma_x0(params)


@pytest.mark.parametrize("material", ["liquid_hydrogen_H2", "lithium_hydride_MICE"])
def test_moliere_screening(material):
    """Moliere screening keeps the nuclear variance and the cutoff b, with a ~1.7x
    larger screening angle and correspondingly fewer scatters"""
    params = make_params(material, 1.0 * u.mm)
    jk = mixture_parameters(params, screening="jackson")
    ml = mixture_parameters(params, screening="moliere")
    assert float(ml.sigma_total) == pytest.approx(float(jk.sigma_total), rel=1e-12)
    f = float(ml.few_a) / float(jk.few_a)
    assert 1.6 < f < 1.9
    assert float(ml.rho) * float(ml.few_a) == pytest.approx(
        float(jk.rho) * float(jk.few_a), rel=1e-12
    )
    var = lambda mp: mp.mean_scatters * mp.few_a**2 * (jnp.log(mp.rho) - 0.5)
    assert float(var(ml)) == pytest.approx(float(var(jk)), rel=1e-9)
    assert float(ml.mean_scatters) < float(jk.mean_scatters)


@pytest.mark.parametrize("screening", ["jackson", "moliere"])
@pytest.mark.parametrize("thickness", [0.1 * u.mm, 1.0 * u.mm])
def test_single_segment_pathwise_gradient(screening, thickness):
    """d<theta_x^2>/dp from pathwise gradients of samples matches the derivative
    of the model variance sigma_T^2 + sigma_e^2 (FEW and SEMI_GAUSSIAN regimes)"""

    def params_at(pz):
        return make_params("liquid_hydrogen_H2", thickness, pz)

    def variance(pz):
        mp = mixture_parameters(params_at(pz), screening)
        return mp.sigma_total**2 + mp.sigma_electron**2

    def sq(pz, k):
        return mixture_scattering_sampler(params_at(pz), k, screening)[0] ** 2

    p0 = 200.0 * u.MeV
    exact = float(jax.grad(variance)(p0))
    keys = jr.split(jr.key(SEED), 200_000)
    pathwise = float(
        jnp.mean(jax.jit(jax.vmap(jax.grad(sq), in_axes=(None, 0)))(p0, keys))
    )
    assert exact < 0.0
    assert pathwise == pytest.approx(exact, rel=0.35)
