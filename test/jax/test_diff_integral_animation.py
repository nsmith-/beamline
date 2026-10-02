"""Illustration of differentiating a Monte Carlo integral w.r.t. a distribution parameter.

We estimate I(mu) = E[f(x)] for x ~ Normal(mu, 1) from samples, and compare
two estimators of dI/dmu at mu = 0:

- pathwise (reparameterization): x_i = mu + Phi^-1(u_i), differentiate each sample.
  For a hard window f the per-sample derivative is zero almost everywhere, so
  autodiff returns exactly 0; for a smooth f it is unbiased, with per-sample terms
  f'(x_i) that grow as f gets narrower.
- weighted (score function): keep x_i fixed, weight them by p(x_i; mu) / p(x_i; 0).
  The derivative lives in the weights, d w_i / d mu = (x_i - mu) / sigma^2.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

import beamline.jax  # noqa: F401  (enables float64)

NSAMPLES = 400
NSHOW = 15
NTRIALS = 2000
WINDOW = (0.5, 2.0)
GAUSS_CENTER = 1.0
GAUSS_WIDTH = 0.2
WIDE_GAUSS_WIDTH = 1.5
MU1 = 0.1
SMOOTH = 0.1
EDGES = np.arange(-3.5, 3.5 + 1e-9, 0.25)
ARROW_DMU = 0.5
"""Arrows in the top left panel show the change for this step in mu"""
BIN_ARROW_DMU = 0.25
"""Arrows on histogram bins show the change for this step in mu"""


def window(x):
    return ((x > WINDOW[0]) & (x < WINDOW[1])).astype(x.dtype)


def soft_window(x, h=SMOOTH):
    return jax.nn.sigmoid((x - WINDOW[0]) / h) - jax.nn.sigmoid((x - WINDOW[1]) / h)


def gaussian(x, width=GAUSS_WIDTH):
    return jnp.exp(-0.5 * ((x - GAUSS_CENTER) / width) ** 2)


def gaussian_integral(mu, width=GAUSS_WIDTH):
    """E[gaussian(x)] for x ~ Normal(mu, 1)"""
    var = 1 + width**2
    return width / jnp.sqrt(var) * jnp.exp(-0.5 * (GAUSS_CENTER - mu) ** 2 / var)


@dataclass(frozen=True)
class Case:
    name: str
    f: Callable
    integral: Callable
    description: str
    """Shown in the histogram title and integral axis label"""
    pathwise_prose: tuple[str, ...]

    def slope(self, mu):
        return jax.grad(self.integral)(mu)


WINDOW_CASE = Case(
    name="window",
    f=window,
    integral=lambda mu: norm.cdf(WINDOW[1] - mu) - norm.cdf(WINDOW[0] - mu),
    description=f"$f = 1[{WINDOW[0]} < x < {WINDOW[1]}]$",
    pathwise_prose=(
        "Samples move (red arrows, top left: $\\partial x_i/\\partial\\mu = 1$)",
        "and carry their weight across bin edges (top right).",
        "But $f$ is a step: $f'(x_i) = 0$ for every sample, so autodiff",
        "returns exactly 0. The gradient is the flux of samples through",
        "the window edges, $p(a) - p(b)$, only seen once the edges are",
        "smoothed (width $h$, $O(h^2)$ bias).",
    ),
)
GAUSS_CASE = Case(
    name="gauss",
    f=gaussian,
    integral=gaussian_integral,
    description=f"$f = e^{{-(x-{GAUSS_CENTER:g})^2/2({GAUSS_WIDTH:g})^2}}$",
    pathwise_prose=(
        "Samples move (red arrows, top left: $\\partial x_i/\\partial\\mu = 1$)",
        "and carry their weight across bin edges (top right).",
        "$f$ is smooth, so each sample contributes $f'(x_i)$: the MC",
        "estimate is smooth in $\\mu$ and autodiff gives an unbiased slope.",
        "But $f' \\sim 1/\\mathrm{width}$, so a narrow $f$ means a few",
        "samples near its peak carry large, noisy contributions.",
    ),
)
CASES = [WINDOW_CASE, GAUSS_CASE]


def reparam_estimate(mu, u, f=window):
    return jnp.mean(f(mu + ndtri(u)))


def weighted_estimate(mu, x0, f=window):
    w = norm.pdf(x0, mu) / norm.pdf(x0, 0.0)
    return jnp.mean(w * f(x0))


def per_sample_gradients(x0, f):
    """Per-sample terms of the pathwise and weighted gradient estimators at mu = 0"""
    pathwise = jax.vmap(jax.grad(f))(x0)
    weighted = f(x0) * x0
    return pathwise, weighted


def histogram(x, w=None):
    counts, _ = np.histogram(x, bins=EDGES, weights=w)
    return counts / (len(x) * np.diff(EDGES))


def _shade(ax, f, xlim):
    """Background shading with opacity proportional to f(x)"""
    y = np.linspace(EDGES[0], EDGES[-1], 1000)
    rgba = np.zeros((len(y), 1, 4))
    rgba[..., :3] = to_rgb("C2")
    rgba[:, 0, 3] = 0.3 * np.asarray(f(jnp.asarray(y)))
    ax.imshow(
        rgba,
        extent=(*xlim, EDGES[0], EDGES[-1]),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )


def _setup_figure(title: str, case: Case):
    fig = plt.figure(figsize=(11, 9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax_q = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_q)
    ax_t = fig.add_subplot(gs[1, 0])
    ax_i = fig.add_subplot(gs[1, 1])
    fig.suptitle(title)

    _shade(ax_q, case.f, (0, 1))
    _shade(ax_h, case.f, (0, 0.6))
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(EDGES[0], EDGES[-1])
    ax_q.set_xlabel("u ~ Uniform(0, 1)")
    ax_q.set_ylabel("x")
    ax_q.set_title(r"inverse CDF $x = \mu + \Phi^{-1}(u)$")
    ax_h.set_xlim(0, 0.6)
    ax_h.set_xlabel("density")
    ax_h.set_title(f"histogram, shading: {case.description}")
    ax_h.tick_params(labelleft=False)

    uu = np.linspace(1e-4, 1 - 1e-4, 400)
    for mu, c in ((0.0, "C0"), (MU1, "C1")):
        ax_q.plot(uu, mu + ndtri(uu), color=c, lw=1, alpha=0.6, label=rf"$\mu$={mu}")
    ax_q.legend(loc="upper left")

    mus = np.linspace(-0.1, 0.2, 301)
    ax_i.plot(mus, case.integral(mus), color="gray", lw=1, label="exact I(μ)")
    ax_i.set_xlabel(r"$\mu$")
    ax_i.set_ylabel(r"$I(\mu) = E[f(x)]$")
    ax_i.set_title("integral vs. mean")
    ax_t.axis("off")
    return fig, ax_q, ax_h, ax_t, ax_i, mus


def _slope_arrow(ax, mu0, value, slope, color, label, dmu=0.08):
    ax.annotate(
        "",
        xy=(mu0 + dmu, value + dmu * slope),
        xytext=(mu0, value),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2},
    )
    ax.plot([], [], color=color, lw=2, label=label)


def _explain(ax, formulas, prose):
    """Formulas need extra vertical room for fractions and sums"""
    y = 1.0
    for line in formulas:
        ax.text(0.0, y, line, transform=ax.transAxes, va="top", fontsize=12)
        y -= 0.11
    ax.text(
        0.0,
        y - 0.03,
        "\n".join(prose),
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        linespacing=1.6,
    )


def _frame_mus(nframes=40):
    t = np.linspace(0, 1, nframes, endpoint=False)
    return 0.5 * MU1 * (1 - np.cos(2 * np.pi * t))


@pytest.fixture(scope="module")
def samples():
    u = jax.random.uniform(jax.random.key(1234), (NSAMPLES,))
    # samples to draw as lines: evenly spaced in quantile
    show = jnp.argsort(u)[jnp.linspace(5, NSAMPLES - 6, NSHOW).astype(int)]
    return u, show


def test_gradient_estimators(samples):
    u, _ = samples
    x0 = ndtri(u)

    # hard window: pathwise derivative is exactly zero
    exact = WINDOW_CASE.slope(0.0)
    assert jax.grad(reparam_estimate)(0.0, u) == 0.0
    # smoothed window: biased by O(h^2) but nonzero, within stat. error
    g_soft = jax.grad(reparam_estimate)(0.0, u, soft_window)
    g_weight = jax.grad(weighted_estimate)(0.0, x0)
    err_weight = jnp.std(x0 * window(x0)) / jnp.sqrt(NSAMPLES)
    assert abs(g_weight - exact) < 3 * err_weight
    assert abs(g_soft - exact) < 3 * err_weight

    # smooth f: both estimators are unbiased
    exact = GAUSS_CASE.slope(0.0)
    pathwise, weighted = per_sample_gradients(x0, gaussian)
    g_path = jax.grad(reparam_estimate)(0.0, u, gaussian)
    g_weight = jax.grad(weighted_estimate)(0.0, x0, gaussian)
    assert g_path == pytest.approx(jnp.mean(pathwise))
    assert g_weight == pytest.approx(jnp.mean(weighted))
    assert abs(g_path - exact) < 3 * jnp.std(pathwise) / jnp.sqrt(NSAMPLES)
    assert abs(g_weight - exact) < 3 * jnp.std(weighted) / jnp.sqrt(NSAMPLES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reparam_animation(samples, artifacts_dir, case: Case):
    u, show = samples
    x0 = ndtri(u)
    fig, ax_q, ax_h, ax_t, ax_i, mus = _setup_figure(
        r"Pathwise (reparameterization) gradient: move the samples, $\partial x_i/\partial\mu$",
        case,
    )

    # top left: each sample moves with dx/dmu = 1 at fixed u
    us, xs = np.asarray(u[show]), np.asarray(x0[show])
    lines = LineCollection([], colors="k", lw=0.8, alpha=0.7)
    ax_q.add_collection(lines)
    (dots,) = ax_q.plot(us, xs, "k.", ms=6)
    dxdmu = jax.vmap(jax.grad(lambda mu, u: mu + ndtri(u)), (None, 0))(0.0, u[show])
    q_samp = ax_q.quiver(
        us,
        xs,
        np.zeros(NSHOW),
        ARROW_DMU * np.asarray(dxdmu),
        color="C3",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
    )
    (curve,) = ax_q.plot([], [], "k", lw=1.5)
    uu = np.linspace(1e-4, 1 - 1e-4, 400)

    # top right: histogram; samples flow between bins as they move
    stairs = ax_h.stairs(
        histogram(x0), EDGES, orientation="horizontal", color="k", lw=1.5
    )
    ax_h.stairs(histogram(x0 + MU1), EDGES, orientation="horizontal", color="C1")
    ax_h.stairs(histogram(x0), EDGES, orientation="horizontal", color="C0")

    # bottom right: MC estimate with common random numbers
    # (a staircase for the hard window, smooth otherwise)
    est = jax.vmap(reparam_estimate, (0, None, None))(mus, u, case.f)
    ax_i.plot(mus, est, color="k", lw=1, drawstyle="steps-post", label="MC estimate")
    ax_i.plot(
        [0, MU1],
        [reparam_estimate(0.0, u, case.f), reparam_estimate(MU1, u, case.f)],
        "o",
        color="k",
        mfc="none",
    )
    (marker,) = ax_i.plot([], [], "ko")
    i0 = float(reparam_estimate(0.0, u, case.f))
    g = float(jax.grad(reparam_estimate)(0.0, u, case.f))
    err = float(jnp.std(per_sample_gradients(x0, case.f)[0]) / jnp.sqrt(NSAMPLES))
    _slope_arrow(ax_i, 0.0, i0, float(case.slope(0.0)), "gray", "exact slope")
    _slope_arrow(ax_i, 0.0, i0, g, "C3", f"grad = {g:.3f} ± {err:.3f}")
    ax_i.legend(loc="upper left", fontsize="small")

    _explain(
        ax_t,
        [
            r"$I(\mu) = \int f(x)\,p(x;\mu)\,dx = \int_0^1 f(\mu + \Phi^{-1}(u))\,du$",
            r"$\hat{I} = \frac{1}{N}\sum_i f(x_i),\quad x_i = \mu + \Phi^{-1}(u_i)$",
            r"$\partial_\mu \hat{I} = \frac{1}{N}\sum_i f'(x_i)\,\partial_\mu x_i$",
        ],
        [
            *case.pathwise_prose,
            "",
            f"N = {NSAMPLES}; arrows show the change for $\\Delta\\mu$ = {ARROW_DMU}",
        ],
    )

    def update(frame: int):
        mu = frame_mus[frame]
        xf = np.asarray(x0) + mu
        curve.set_data(uu, mu + ndtri(uu))
        lines.set_segments(
            [[(ui, xi + mu), (1, xi + mu)] for ui, xi in zip(us, xs, strict=True)]
        )
        dots.set_data(us, xs + mu)
        q_samp.set_offsets(np.stack([us, xs + mu], axis=-1))
        stairs.set_data(histogram(xf), EDGES)
        marker.set_data([mu], [reparam_estimate(mu, u, case.f)])
        return (curve, lines, dots, q_samp, stairs, marker)

    frame_mus = _frame_mus()
    update(0)
    fig.savefig(artifacts_dir / f"reparam_gradient_{case.name}.png", dpi=150)
    anim = FuncAnimation(fig, update, frames=len(frame_mus), blit=True)
    anim.save(
        artifacts_dir / f"reparam_gradient_{case.name}.gif", writer="pillow", fps=10
    )
    plt.close(fig)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_weighted_animation(samples, artifacts_dir, case: Case):
    u, show = samples
    x0 = ndtri(u)
    fig, ax_q, ax_h, ax_t, ax_i, mus = _setup_figure(
        r"Weighted (score function) gradient: fix the samples, $\partial w_i/\partial\mu$",
        case,
    )

    # top left: samples stay at fixed x, their quantile u = F(x; mu) slides left
    # by du/dmu = -p(x).  The weight is the ratio of the curve slopes du/dx.
    us, xs = np.asarray(u[show]), np.asarray(x0[show])
    score = np.asarray(jax.vmap(jax.grad(norm.logpdf, argnums=1), (0, None))(xs, 0.0))
    colors = np.where(score > 0, "C3", "C0")
    lines = LineCollection(
        [[(ui, xi), (1, xi)] for ui, xi in zip(us, xs, strict=True)],
        colors=colors,
        alpha=0.7,
    )
    ax_q.add_collection(lines)
    ax_q.plot(us, xs, ".", color="gray", ms=5)
    (dots,) = ax_q.plot([], [], "k.", ms=6)
    dudmu = jax.vmap(jax.grad(lambda mu, x: norm.cdf(x, mu)), (None, 0))(0.0, xs)
    ax_q.quiver(
        us,
        xs,
        ARROW_DMU * np.asarray(dudmu),
        np.zeros(NSHOW),
        color="k",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
    )
    (curve,) = ax_q.plot([], [], "k", lw=1.5)
    uu = np.linspace(1e-4, 1 - 1e-4, 400)

    # top right: same samples, reweighted; bin derivative is the sum of scores
    def weights(mu):
        return norm.pdf(x0, mu) / norm.pdf(x0, 0.0)

    stairs = ax_h.stairs(
        histogram(x0), EDGES, orientation="horizontal", color="k", lw=1.5
    )
    ax_h.stairs(
        histogram(x0, weights(MU1)), EDGES, orientation="horizontal", color="C1"
    )
    ax_h.stairs(histogram(x0), EDGES, orientation="horizontal", color="C0")
    # np.histogram is not traceable, so contract the weight jacobian by hand
    dw = jax.jacfwd(weights)(0.0)
    dbins = histogram(x0, dw)
    centers = 0.5 * (EDGES[1:] + EDGES[:-1])
    ax_h.quiver(
        histogram(x0),
        centers,
        BIN_ARROW_DMU * dbins,
        np.zeros_like(centers),
        color=np.where(centers > 0, "C3", "C0"),
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
    )

    # bottom right: the reweighted MC estimate is smooth in mu
    est = jax.vmap(weighted_estimate, (0, None, None))(mus, x0, case.f)
    ax_i.plot(mus, est, color="k", lw=1, label="MC estimate (reweighted)")
    ax_i.plot(
        [0, MU1],
        [weighted_estimate(0.0, x0, case.f), weighted_estimate(MU1, x0, case.f)],
        "o",
        color="k",
        mfc="none",
    )
    (marker,) = ax_i.plot([], [], "ko")
    i0 = float(weighted_estimate(0.0, x0, case.f))
    g_w = float(jax.grad(weighted_estimate)(0.0, x0, case.f))
    err_w = float(jnp.std(per_sample_gradients(x0, case.f)[1]) / jnp.sqrt(NSAMPLES))
    _slope_arrow(ax_i, 0.0, i0, float(case.slope(0.0)), "gray", "exact slope")
    _slope_arrow(ax_i, 0.0, i0, g_w, "C3", f"grad = {g_w:.3f} ± {err_w:.3f}")
    ax_i.legend(loc="upper left", fontsize="small")

    _explain(
        ax_t,
        [
            r"$I(\mu) = \int f(x)\,\frac{p(x;\mu)}{p(x;\mu_0)}\,p(x;\mu_0)\,dx$",
            r"$\hat{I} = \frac{1}{N}\sum_i w_i(\mu) f(x_i),\quad x_i \sim p(x;\mu_0)$",
            (
                r"$\partial_\mu \hat{I} = \frac{1}{N}\sum_i f(x_i)\,\partial_\mu \log p(x_i;\mu)"
                r" = \frac{1}{N}\sum_i f(x_i)\,\frac{x_i-\mu}{\sigma^2}$"
            ),
        ],
        [
            "Samples stay put. As the curve shifts, each sample's quantile",
            r"slides by $\partial u/\partial\mu = -p(x_i)$ (arrows, top left) and its",
            "weight is the ratio of curve slopes $du/dx$ (line width, $w^4$).",
            "Bins change height, not membership: red/blue = weight up/down.",
            "$f$ is never differentiated, so its edges or width don't matter;",
            "the noise comes from the score $(x_i - \\mu)/\\sigma^2$ instead.",
            "",
            (
                f"N = {NSAMPLES}; arrows show the change for $\\Delta\\mu$ = "
                f"{ARROW_DMU} (top left), {BIN_ARROW_DMU} (bins)"
            ),
        ],
    )

    def update(frame: int):
        mu = frame_mus[frame]
        w = np.asarray(weights(mu))
        curve.set_data(uu, mu + ndtri(uu))
        dots.set_data(norm.cdf(xs, mu), xs)
        lines.set_linewidth(1.5 * np.asarray(norm.pdf(xs, mu) / norm.pdf(xs, 0.0)) ** 4)
        stairs.set_data(histogram(x0, w), EDGES)
        marker.set_data([mu], [weighted_estimate(mu, x0, case.f)])
        return (curve, dots, lines, stairs, marker)

    frame_mus = _frame_mus()
    update(0)
    fig.savefig(artifacts_dir / f"weighted_gradient_{case.name}.png", dpi=150)
    anim = FuncAnimation(fig, update, frames=len(frame_mus), blit=True)
    anim.save(
        artifacts_dir / f"weighted_gradient_{case.name}.gif", writer="pillow", fps=10
    )
    plt.close(fig)


def _trial_gradients(width, key):
    """Pathwise and weighted gradient estimates from NTRIALS independent sample sets"""

    def trial(key):
        x0 = jax.random.normal(key, (NSAMPLES,))
        pathwise, weighted = per_sample_gradients(x0, lambda x: gaussian(x, width))
        return jnp.mean(pathwise), jnp.mean(weighted)

    return jax.vmap(trial)(jax.random.split(key, NTRIALS))


def test_gradient_variance(artifacts_dir):
    """Repeated trials at a fixed sample budget: which estimator is less noisy?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    fig.suptitle(
        f"Gradient estimates of $\\partial_\\mu E[f(x)]$ at $\\mu=0$, "
        f"{NTRIALS} trials of N = {NSAMPLES} samples each, "
        f"$f = e^{{-(x-{GAUSS_CENTER:g})^2/2w^2}}$"
    )
    key = jax.random.key(42)
    for ax, width in zip(axes[:2], (GAUSS_WIDTH, WIDE_GAUSS_WIDTH), strict=True):
        pathwise, weighted = _trial_gradients(width, key)
        exact = float(jax.grad(gaussian_integral)(0.0, width))
        lo = min(jnp.min(pathwise), jnp.min(weighted))
        hi = max(jnp.max(pathwise), jnp.max(weighted))
        bins = np.linspace(lo, hi, 60)
        for est, label, c in (
            (pathwise, "pathwise", "C3"),
            (weighted, "weighted", "C0"),
        ):
            ax.hist(
                est,
                bins=bins,
                histtype="stepfilled",
                alpha=0.35,
                color=c,
                label=f"{label}: {jnp.mean(est):.3f} ± {jnp.std(est):.3f}",
            )
            ax.hist(est, bins=bins, histtype="step", color=c)
        ax.axvline(exact, color="k", ls="--", label=f"exact = {exact:.3f}")
        ax.set_title(f"width w = {width}")
        ax.set_xlabel(r"estimated $\partial_\mu I$")
        ax.set_ylabel("trials")
        ax.legend(fontsize="small")

        # same budget, same expectation; which one wins depends on the width
        assert jnp.mean(pathwise) == pytest.approx(exact, abs=0.01)
        assert jnp.mean(weighted) == pytest.approx(exact, abs=0.01)
        if width < 0.5:
            assert jnp.std(weighted) < jnp.std(pathwise)
        else:
            assert jnp.std(pathwise) < jnp.std(weighted)

    # std of a single N-sample estimate vs width of f, from a large sample
    ax = axes[2]
    x = jax.random.normal(jax.random.key(7), (1_000_000,))
    widths = np.geomspace(0.05, 3.0, 40)
    stds = np.array(
        [
            [jnp.std(g) for g in per_sample_gradients(x, partial(gaussian, width=w))]
            for w in widths
        ]
    ) / np.sqrt(NSAMPLES)
    ax.plot(widths, stds[:, 0], color="C3", label="pathwise")
    ax.plot(widths, stds[:, 1], color="C0", label="weighted")
    for w in (GAUSS_WIDTH, WIDE_GAUSS_WIDTH):
        ax.axvline(w, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width w of f")
    ax.set_ylabel(f"std of gradient estimate (N = {NSAMPLES})")
    ax.set_title(r"pathwise wins once f is wider than ~0.75$\sigma$")
    ax.legend()
    fig.savefig(artifacts_dir / "gradient_variance_gauss.png", dpi=150)
    plt.close(fig)
