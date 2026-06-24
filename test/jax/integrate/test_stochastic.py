"""Tests for stochastic energy-loss propagation (``stochastic_solve``).

Covers an end-to-end ensemble propagation through an absorber, the value-gradient
(pathwise) check in both AD modes, and the weight-gradient (score-function)
estimator. Artifacts (figures, CSV) are written under ``test_artifacts/``.
"""

import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.straggling import (
    landau_energy_loss_sampler,
    landau_energy_loss_sampler_wg,
)
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField
from beamline.jax.integrate.stochastic import stochastic_solve
from beamline.jax.kinematics import MuonStateDz

ABSORBER_RADIUS = 100.0 * u.mm
ABSORBER_LENGTH = 100.0 * u.mm


def make_absorber(char_length: float = 10.0 * u.mm) -> AbsorberCylinder:
    return AbsorberCylinder(
        material=MATERIALS["lithium_hydride_LiH"],
        radius=ABSORBER_RADIUS,
        length=ABSORBER_LENGTH,
        char_length=char_length,
    )


def make_muon(pz: float = 200.0 * u.MeV, x: float = 0.0) -> MuonStateDz:
    return MuonStateDz.make(
        position=Cartesian4.make(x=x, z=-200.0 * u.mm),
        momentum=Cartesian3.make(z=pz),
        q=1,
    )


def _free_field() -> SimpleEMField:
    return SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())


def _save_grid() -> jnp.ndarray:
    return jnp.linspace(-200.0 * u.mm, 200.0 * u.mm, 5)


def test_stochastic_propagation(artifacts_dir):
    """An ensemble loses ~Bethe-Bloch energy through the absorber, with spread."""
    field = _free_field()
    absorber = make_absorber()
    start = make_muon()
    zs = _save_grid()

    keys = jr.split(jr.key(0), 1000)
    run = jax.jit(
        jax.vmap(lambda k: stochastic_solve(field, absorber, start, zs, k)[0])
    )
    ys = run(keys)

    energy_initial = float(start.kin.t.ct)
    energy_final = np.asarray(ys.kin.t.ct[:, -1])
    loss = energy_initial - energy_final

    assert (energy_final < energy_initial).all()  # everyone loses energy
    assert loss.std() > 0.0  # straggling produces a spread

    # The ensemble-mean loss should track the Bethe-Bloch mean for the full
    # traversal (the absorber length on-axis).
    reference = float(
        absorber.interaction_params(start, ABSORBER_LENGTH).mean_energy_loss
    )
    assert loss.mean() == pytest.approx(reference, rel=0.4)

    # A muon that misses the absorber must conserve energy exactly.
    miss, _ = jax.jit(lambda s, k: stochastic_solve(field, absorber, s, zs, k))(
        make_muon(x=200.0 * u.mm), jr.key(1)
    )
    assert float(miss.kin.t.ct[-1]) == pytest.approx(energy_initial, rel=1e-9)

    with open(artifacts_dir / "stochastic_final_states.csv", "w") as f:
        f.write("xf,yf,zf,Ef,loss\n")
        for i in range(len(energy_final)):
            f.write(
                f"{float(ys.kin.p.x[i, -1]) / u.mm:.6f},"
                f"{float(ys.kin.p.y[i, -1]) / u.mm:.6f},"
                f"{float(ys.kin.p.z[i, -1]) / u.mm:.6f},"
                f"{energy_final[i] / u.MeV:.6f},"
                f"{loss[i] / u.MeV:.6f}\n"
            )

    fig, ax = plt.subplots()
    ax.hist(loss / u.MeV, bins=40)
    ax.axvline(reference / u.MeV, color="k", ls="--", label="Bethe-Bloch mean")
    ax.axvline(loss.mean() / u.MeV, color="C1", ls="-", label="ensemble mean")
    ax.set_xlabel("energy loss [MeV]")
    ax.set_ylabel("count")
    ax.set_title("Straggling through 100 mm LiH (dummy sampler)")
    ax.legend()
    fig.savefig(artifacts_dir / "stochastic_energy_loss.png", dpi=150)
    plt.close(fig)


def _mean_final_energy(forward_mode, pz):
    field = _free_field()
    absorber = make_absorber()
    zs = _save_grid()
    start = make_muon(pz)
    ys = jax.vmap(
        lambda k: stochastic_solve(
            field, absorber, start, zs, k, forward_mode=forward_mode
        )[0]
    )(jr.split(jr.key(1), 256))
    return jnp.mean(ys.kin.t.ct[:, -1])


@pytest.mark.parametrize("forward_mode", [False, True])
def test_stochastic_gradient(forward_mode):
    """Gradient of mean final energy flows through the kicks, in either AD mode.

    Reverse-mode (``forward_mode=False``) is the default for scalar-loss
    optimization; forward-mode is exercised too since the reparameterized sampler
    supports both. Both are checked against a central finite difference (the
    pathwise sampler makes a common-random-number finite difference exact to
    tolerance).
    """
    pz0 = 200.0 * u.MeV
    if forward_mode:
        _, grad = jax.jvp(lambda pz: _mean_final_energy(True, pz), (pz0,), (1.0,))
        grad = float(grad)
    else:
        grad = float(jax.grad(lambda pz: _mean_final_energy(False, pz))(pz0))
    assert jnp.isfinite(grad)

    h = 1e-2 * u.MeV
    fd = float(
        (
            _mean_final_energy(forward_mode, pz0 + h)
            - _mean_final_energy(forward_mode, pz0 - h)
        )
        / (2 * h)
    )
    assert grad == pytest.approx(fd, rel=1e-3)


def _weighted_mean_final_energy(pz, sampler, n=256):
    """Self-normalized weighted ensemble mean of the final energy.

    For the value samplers the log-weights are zero, so this is the plain mean;
    for the weight sampler the score-function gradient enters through the weights.
    """
    field = _free_field()
    absorber = make_absorber()
    zs = _save_grid()
    start = make_muon(pz)

    def one(k):
        ys, stats = stochastic_solve(field, absorber, start, zs, k, sampler=sampler)
        return ys.kin.t.ct[-1], stats["log_weight"]

    energy_final, log_weight = jax.vmap(one)(jr.split(jr.key(2), n))
    weights = jnp.exp(log_weight - logsumexp(log_weight))
    return jnp.sum(weights * energy_final)


def test_stochastic_weight_plumbing():
    """The WG sampler's log-weight is threaded through the solver consistently.

    The WG sampler draws from a ``stop_gradient``-ed reference at the sampling
    parameters, so its accumulated ``log_weight`` is numerically zero and its
    forward result is identical to the value-gradient (SG) sampler; the two differ
    only under differentiation. This checks the plumbing (the per-step weight is
    accumulated and returned in ``stats``) and the forward equivalence.

    The *gradient* of the score-function estimator is validated at the sampler
    level in ``test/jax/absorber/test_straggling.py``. It is not asserted through
    full propagation here: the unbounded Landau tail can drive a single segment's
    energy loss large enough to floor the longitudinal momentum, and the
    ``MuonStateDz`` right-hand side's ``1/p_z`` then makes the *reverse-mode*
    gradient diverge (the forward pass survives in float64). Stabilizing this needs
    Landau tail truncation and/or a stopping condition -- tracked as follow-up.
    """
    pz0 = 200.0 * u.MeV

    # The accumulated weight is numerically zero (it carries only a gradient).
    _, stats = jax.vmap(
        lambda k: stochastic_solve(
            _free_field(),
            make_absorber(),
            make_muon(pz0),
            _save_grid(),
            k,
            sampler=landau_energy_loss_sampler_wg,
        )
    )(jr.split(jr.key(2), 256))
    assert "log_weight" in stats
    assert np.allclose(np.asarray(stats["log_weight"]), 0.0, atol=1e-9)

    # WG and SG agree in the forward pass (same draws, unit weights).
    value_wg = float(_weighted_mean_final_energy(pz0, landau_energy_loss_sampler_wg))
    value_sg = float(_weighted_mean_final_energy(pz0, landau_energy_loss_sampler))
    assert value_wg == pytest.approx(value_sg, rel=1e-9)
