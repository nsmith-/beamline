"""Illustration of differentiating a Monte Carlo integral w.r.t. a distribution parameter.

We estimate I(mu) = P(a < x < b) for x ~ Normal(mu, 1) from samples, and compare
two estimators of dI/dmu at mu = 0:

- pathwise (reparameterization): x_i = mu + Phi^-1(u_i), differentiate each sample.
  For a hard window the per-sample derivative is zero almost everywhere, so autodiff
  returns exactly 0; the slope only appears once the window edges are smoothed.
- weighted (score function): keep x_i fixed, weight them by p(x_i; mu) / p(x_i; 0).
  The derivative lives in the weights, d w_i / d mu = (x_i - mu) / sigma^2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import beamline.jax  # noqa: F401  (enables float64)

NSAMPLES = 400
NSHOW = 15
WINDOW = (0.5, 2.0)
MU1 = 0.1
SMOOTH = 0.1
EDGES = np.arange(-3.5, 3.5 + 1e-9, 0.25)
ARROW_DMU = 0.5
"""Arrows in the top panels show the change for this step in mu"""


def window(x):
    return ((x > WINDOW[0]) & (x < WINDOW[1])).astype(x.dtype)


def soft_window(x, h=SMOOTH):
    return jax.nn.sigmoid((x - WINDOW[0]) / h) - jax.nn.sigmoid((x - WINDOW[1]) / h)


def true_integral(mu):
    return norm.cdf(WINDOW[1] - mu) - norm.cdf(WINDOW[0] - mu)


def true_slope(mu):
    return norm.pdf(WINDOW[0] - mu) - norm.pdf(WINDOW[1] - mu)


def reparam_estimate(mu, u, f=window):
    return jnp.mean(f(mu + ndtri(u)))


def weighted_estimate(mu, x0, f=window):
    w = norm.pdf(x0, mu) / norm.pdf(x0, 0.0)
    return jnp.mean(w * f(x0))


def histogram(x, w=None):
    counts, _ = np.histogram(x, bins=EDGES, weights=w)
    return counts / (len(x) * np.diff(EDGES))


def soft_histogram(x):
    """Histogram with sigmoid-smoothed bin edges, so pathwise derivatives exist"""
    lo, hi = jnp.asarray(EDGES[:-1]), jnp.asarray(EDGES[1:])
    s = jax.nn.sigmoid((x[:, None] - lo) / SMOOTH) - jax.nn.sigmoid(
        (x[:, None] - hi) / SMOOTH
    )
    return s.sum(axis=0) / (len(x) * jnp.diff(jnp.asarray(EDGES)))


def _setup_figure(title: str):
    fig = plt.figure(figsize=(11, 9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax_q = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_q)
    ax_t = fig.add_subplot(gs[1, 0])
    ax_i = fig.add_subplot(gs[1, 1])
    fig.suptitle(title)

    for ax in (ax_q, ax_h):
        ax.axhspan(*WINDOW, color="C2", alpha=0.15, lw=0)
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(EDGES[0], EDGES[-1])
    ax_q.set_xlabel("u ~ Uniform(0, 1)")
    ax_q.set_ylabel("x")
    ax_q.set_title(r"inverse CDF $x = \mu + \Phi^{-1}(u)$")
    ax_h.set_xlim(0, 0.6)
    ax_h.set_xlabel("density")
    ax_h.set_title(f"histogram, window {WINDOW[0]} < x < {WINDOW[1]}")
    ax_h.tick_params(labelleft=False)

    uu = np.linspace(1e-4, 1 - 1e-4, 400)
    for mu, c in ((0.0, "C0"), (MU1, "C1")):
        ax_q.plot(uu, mu + ndtri(uu), color=c, lw=1, alpha=0.6, label=rf"$\mu$={mu}")
    ax_q.legend(loc="upper left")

    mus = np.linspace(-0.1, 0.2, 301)
    ax_i.plot(mus, true_integral(mus), color="gray", lw=1, label="exact I(μ)")
    ax_i.set_xlabel(r"$\mu$")
    ax_i.set_ylabel(rf"$I(\mu) = P({WINDOW[0]} < x < {WINDOW[1]})$")
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
    exact = true_slope(0.0)

    # hard window: pathwise derivative is exactly zero
    assert jax.grad(reparam_estimate)(0.0, u) == 0.0
    # smoothed window: biased by O(h^2) but nonzero, within stat. error
    g_soft = jax.grad(reparam_estimate)(0.0, u, soft_window)
    g_weight = jax.grad(weighted_estimate)(0.0, x0)
    err_weight = jnp.std(x0 * window(x0)) / jnp.sqrt(NSAMPLES)
    assert abs(g_weight - exact) < 3 * err_weight
    assert abs(g_soft - exact) < 3 * err_weight


def test_reparam_animation(samples, artifacts_dir):
    u, show = samples
    x0 = ndtri(u)
    fig, ax_q, ax_h, ax_t, ax_i, mus = _setup_figure(
        r"Pathwise (reparameterization) gradient: move the samples, $\partial x_i/\partial\mu$"
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

    # top right: histogram; pathwise bin derivatives need smoothed bin edges
    stairs = ax_h.stairs(
        histogram(x0), EDGES, orientation="horizontal", color="k", lw=1.5
    )
    ax_h.stairs(histogram(x0 + MU1), EDGES, orientation="horizontal", color="C1")
    ax_h.stairs(histogram(x0), EDGES, orientation="horizontal", color="C0")
    dbins = jax.jacfwd(lambda mu: soft_histogram(mu + x0))(0.0)
    centers = 0.5 * (EDGES[1:] + EDGES[:-1])
    ax_h.quiver(
        histogram(x0),
        centers,
        ARROW_DMU * np.asarray(dbins),
        np.zeros_like(centers),
        color="C3",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
    )

    # bottom right: MC estimate (with common random numbers) is a staircase
    est = jax.vmap(reparam_estimate, (0, None))(mus, u)
    ax_i.plot(mus, est, color="k", lw=1, drawstyle="steps-post", label="MC estimate")
    ax_i.plot(
        [0, MU1],
        [reparam_estimate(0.0, u), reparam_estimate(MU1, u)],
        "o",
        color="k",
        mfc="none",
    )
    (marker,) = ax_i.plot([], [], "ko")
    i0 = float(reparam_estimate(0.0, u))
    g_hard = float(jax.grad(reparam_estimate)(0.0, u))
    g_soft = float(jax.grad(reparam_estimate)(0.0, u, soft_window))
    _slope_arrow(ax_i, 0.0, i0, float(true_slope(0.0)), "gray", "exact slope")
    _slope_arrow(ax_i, 0.0, i0, g_hard, "C3", f"grad, hard window = {g_hard:.3f}")
    _slope_arrow(
        ax_i, 0.0, i0, g_soft, "C2", f"grad, smoothed (h={SMOOTH}) = {g_soft:.3f}"
    )
    ax_i.legend(loc="upper left", fontsize="small")

    _explain(
        ax_t,
        [
            r"$I(\mu) = \int f(x)\,p(x;\mu)\,dx = \int_0^1 f(\mu + \Phi^{-1}(u))\,du$",
            r"$\hat{I} = \frac{1}{N}\sum_i f(x_i),\quad x_i = \mu + \Phi^{-1}(u_i)$",
            r"$\partial_\mu \hat{I} = \frac{1}{N}\sum_i f'(x_i)\,\partial_\mu x_i$",
        ],
        [
            "Samples move (red arrows, top left: $\\partial x_i/\\partial\\mu = 1$)",
            "and carry their weight across bin edges (red arrows,",
            "top right, smoothed edges). But $f$ is a step: $f'(x_i) = 0$",
            "for every sample, so autodiff returns exactly 0. The gradient",
            "is the flux of samples through the window edges, $p(a) - p(b)$,",
            "only seen once the edges are smoothed (width $h$, $O(h^2)$ bias).",
            "",
            (
                f"N = {NSAMPLES}; arrows in top panels show the change for "
                f"$\\Delta\\mu$ = {ARROW_DMU}"
            ),
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
        marker.set_data([mu], [reparam_estimate(mu, u)])
        return (curve, lines, dots, q_samp, stairs, marker)

    frame_mus = _frame_mus()
    update(0)
    fig.savefig(artifacts_dir / "reparam_gradient.png", dpi=150)
    anim = FuncAnimation(fig, update, frames=len(frame_mus), blit=True)
    anim.save(artifacts_dir / "reparam_gradient.gif", writer="pillow", fps=10)
    plt.close(fig)


def test_weighted_animation(samples, artifacts_dir):
    u, show = samples
    x0 = ndtri(u)
    fig, ax_q, ax_h, ax_t, ax_i, mus = _setup_figure(
        r"Weighted (score function) gradient: fix the samples, $\partial w_i/\partial\mu$"
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
        ARROW_DMU * dbins,
        np.zeros_like(centers),
        color=np.where(centers > 0, "C3", "C0"),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
    )

    # bottom right: the reweighted MC estimate is smooth in mu
    est = jax.vmap(weighted_estimate, (0, None))(mus, x0)
    ax_i.plot(mus, est, color="k", lw=1, label="MC estimate (reweighted)")
    ax_i.plot(
        [0, MU1],
        [weighted_estimate(0.0, x0), weighted_estimate(MU1, x0)],
        "o",
        color="k",
        mfc="none",
    )
    (marker,) = ax_i.plot([], [], "ko")
    i0 = float(weighted_estimate(0.0, x0))
    g_w = float(jax.grad(weighted_estimate)(0.0, x0))
    err_w = float(jnp.std(x0 * window(x0)) / jnp.sqrt(NSAMPLES))
    _slope_arrow(ax_i, 0.0, i0, float(true_slope(0.0)), "gray", "exact slope")
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
            "No boundary terms needed, $f$ is never differentiated,",
            "at the cost of higher variance.",
            "",
            (
                f"N = {NSAMPLES}; arrows in top panels show the change for "
                f"$\\Delta\\mu$ = {ARROW_DMU}"
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
        marker.set_data([mu], [weighted_estimate(mu, x0)])
        return (curve, dots, lines, stairs, marker)

    frame_mus = _frame_mus()
    update(0)
    fig.savefig(artifacts_dir / "weighted_gradient.png", dpi=150)
    anim = FuncAnimation(fig, update, frames=len(frame_mus), blit=True)
    anim.save(artifacts_dir / "weighted_gradient.gif", writer="pillow", fps=10)
    plt.close(fig)
