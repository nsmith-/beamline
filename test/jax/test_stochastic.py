"""Tests for stochastic energy-loss propagation scaffolding.

Covers the material volume geometry, the (dummy) energy-loss sampler, an
end-to-end ensemble propagation through an absorber, and a reverse-mode gradient
check. Artifacts (figures, CSV) are written under ``test_artifacts/``.
"""

import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from matplotlib import pyplot as plt

from beamline.jax.absorber.material import MATERIALS, StragglingParams
from beamline.jax.absorber.straggling import dummy_energy_loss_sampler
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
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


def test_absorber_geometry():
    """contains/signed_distance match the expected cylindrical bounds."""
    absorber = make_absorber()

    assert bool(absorber.contains(Cartesian3.make()))  # origin, inside
    assert not bool(absorber.contains(Cartesian3.make(z=100.0 * u.mm)))  # past end
    assert not bool(absorber.contains(Cartesian3.make(x=150.0 * u.mm)))  # past radius

    # On-axis ray approaching the front face (at z = -length/2 = -50 mm) from
    # z = -200 mm at unit speed: the nearest surface is 150 mm ahead.
    entering = Tangent(p=Cartesian3.make(z=-200.0 * u.mm), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_distance(entering)) == pytest.approx(
        150.0 * u.mm, rel=1e-6
    )

    # A ray offset beyond the radius, parallel to the axis, never enters.
    missing = Tangent(
        p=Cartesian3.make(x=150.0 * u.mm, z=-200.0 * u.mm), t=Cartesian3.make(z=1.0)
    )
    assert jnp.isinf(absorber.signed_distance(missing))


def test_dummy_sampler_statistics(artifacts_dir):
    """The dummy sampler is one-sided with the prescribed Bethe-Bloch mean."""
    mean = 2.0 * u.MeV
    params = StragglingParams(
        xi=0.1 * u.MeV, kappa=0.5, mean_energy_loss=mean, mode_energy_loss=1.5 * u.MeV
    )
    keys = jr.split(jr.key(0), 50_000)
    samples = np.asarray(jax.vmap(lambda k: dummy_energy_loss_sampler(params, k))(keys))

    assert (samples > 0).all()  # one-sided (energy is only lost)
    assert samples.mean() == pytest.approx(mean, rel=0.03)

    fig, ax = plt.subplots()
    ax.hist(samples / u.MeV, bins=80)
    ax.axvline(mean / u.MeV, color="k", ls="--", label="mean")
    ax.set_xlabel("energy loss [MeV]")
    ax.set_ylabel("count")
    ax.set_title("Dummy energy-loss sampler (sum of two exponentials)")
    ax.legend()
    fig.savefig(artifacts_dir / "dummy_sampler_hist.png", dpi=150)
    plt.close(fig)


def test_stochastic_propagation(artifacts_dir):
    """An ensemble loses ~Bethe-Bloch energy through the absorber, with spread."""
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    absorber = make_absorber()
    start = make_muon()
    zs = jnp.linspace(-200.0 * u.mm, 200.0 * u.mm, 5)

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
    field = SimpleEMField(E0=Cartesian3.make(), B0=Cartesian3.make())
    absorber = make_absorber()
    zs = jnp.linspace(-200.0 * u.mm, 200.0 * u.mm, 5)
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
    supports both. Both are checked against a central finite difference.
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
