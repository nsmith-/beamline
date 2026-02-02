import hepunits as u
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from matplotlib import pyplot as plt

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.emfield import SimpleEMField, particle_interaction
from beamline.jax.kinematics import MuonState, ParticleState


def test_interaction(artifacts_dir):
    field = SimpleEMField(
        E0=Cartesian3(coords=jnp.array([0.0, 0.0, 0.0])),
        B0=Cartesian3(coords=jnp.array([0.0, 0.0, 1.0]) * u.tesla),
    )

    def func(_ct, state: ParticleState, _args) -> ParticleState:
        return particle_interaction(state, field)

    term = ODETerm(func)
    solver = Dopri5()
    start = MuonState.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(x=10 * u.MeV),
        q=1,
    )
    t0, t1 = 0.0 * u.m, 10.0 * u.m
    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / 100,
        y0=start,
        saveat=SaveAt(ts=jnp.linspace(t0, t1, 30)),
    )
    # raise RuntimeError(eqx.tree_pformat(sol, short_arrays=False))

    fig, ax = plt.subplots()

    res: MuonState = sol.ys

    ax.plot(
        res.kin.point.x.coords[:, 0] / u.mm,
        res.kin.point.x.coords[:, 1] / u.mm,
    )
    fig.savefig(artifacts_dir / "test_emfield_interaction.png")
