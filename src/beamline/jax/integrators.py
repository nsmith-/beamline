import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint


@dataclass(frozen=True)
class IntegratorConfig:
    c: tuple[float]
    d: tuple[float]
    dt: float

    def __post_init__(self):
        assert len(self.c) == len(self.d)
        assert jnp.allclose(sum(self.c), 1)
        assert jnp.allclose(sum(self.d), 1)
        cond2a = sum(c * sum(self.d[i:]) for i, c in enumerate(self.c))
        cond2b = sum(c * sum(self.d[:i]) for i, c in enumerate(self.c))
        if not jnp.allclose(cond2a, 0.5) or not jnp.allclose(cond2b, 0.5):
            warnings.warn(
                f"Failed to satisfy the symplectic condition at second order for {self}",
                stacklevel=2,
            )
        seq = []
        for ci, di in zip(reversed(self.c), reversed(self.d)):
            if di != 0.0:
                seq.append(di)
            if ci != 0.0:
                seq.append(ci)
        if list(reversed(seq)) != seq:
            warnings.warn(
                f"Failed to satisfy symmetric condition for {self}", stacklevel=2
            )

    def steps(self):
        seq = [(0.0, 0.0)]
        for ci, di in zip(reversed(self.c), reversed(self.d)):
            if di != 0.0:
                seq.append((seq[-1][0], seq[-1][1] + di))
            if ci != 0.0:
                seq.append((seq[-1][0] + ci, seq[-1][1]))
        return jnp.array(seq)


def Omega(co_state):
    """Symplectic form inverse (maps cotangent space to tangent space)"""
    return jnp.array((co_state[1], -co_state[0]))


@partial(jax.jit, static_argnums=(0, 3))
def symplectic_integrator(
    hamiltonian: Callable, state0, times, config: IntegratorConfig
):
    """Symplectic integrator for separable Hamiltonian systems

    We assume Hamiltonian is separable, i.e. H(q, p) = T(p) + V(q)
    and integrate it using a split-step symplectic integrator.

    TODO: time-dependent Hamiltonians
    TODO: adaptive time-stepping https://arxiv.org/pdf/1108.0322
    """
    H0 = hamiltonian(state0)
    q0, p0 = state0
    dHdq = jax.jacobian(lambda q, p: hamiltonian((q, p)), argnums=0)
    dHdp = jax.jacobian(lambda q, p: hamiltonian((q, p)), argnums=1)

    cdstack = jnp.stack([jnp.array(config.c), jnp.array(config.d)], axis=1)

    def symplectic_step(q, p, dt):
        # https://en.wikipedia.org/wiki/Symplectic_integrator or
        # Eqn. 7 https://fse.studenttheses.ub.rug.nl/20185/1/bMATH_2019_PimJC.pdf
        def inner_loop(qp, cd):
            q, p = qp
            ci, di = cd
            p = p - di * dt * dHdq(q, p)
            q = q + ci * dt * dHdp(q, p)
            return (q, p), None

        (q, p), _ = jax.lax.scan(inner_loop, (q, p), cdstack, reverse=True)
        return q, p

    def body_fn(state, tnext):
        q, p, t = state

        def cond_fn(qpt):
            _, _, t = qpt
            return t + config.dt < tnext

        def loop_fn(qpt):
            q, p, t = qpt
            q, p = symplectic_step(q, p, config.dt)
            return q, p, t + config.dt

        q, p, t = jax.lax.while_loop(cond_fn, loop_fn, (q, p, t))
        q, p = symplectic_step(q, p, tnext - t)
        return (q, p, tnext), (q, p, hamiltonian((q, p)))

    _, (q, p, H) = jax.lax.scan(body_fn, (q0, p0, times[0]), times[1:])
    return (
        jnp.concatenate((q0[None], q)),
        jnp.concatenate((p0[None], p)),
        jnp.concatenate((H0[None], H)),
    )


@partial(jax.jit, static_argnums=(0,))
def flow_integrator(hamiltonian: Callable, state0, times):
    H0 = hamiltonian(state0)

    def hflow(state):
        return Omega(jax.jacobian(hamiltonian)(state))

    def body_fn(carry, tnext):
        state, t = carry
        dt = tnext - t
        flow = hflow(state)
        flow2 = jax.jvp(hflow, (state,), (flow,))[1]
        state = state + dt * flow + 0.5 * dt**2 * flow2
        return (state, tnext), (state, hamiltonian(state), jnp.sum(flow2 * flow))

    _, (track, H, flow) = jax.lax.scan(body_fn, (state0, times[0]), times[1:])
    track = jnp.concatenate((state0[None], track))
    return (
        track[:, 0],
        track[:, 1],
        jnp.concatenate((H0[None], H)),
        jnp.concatenate((jnp.zeros_like(flow[0])[None], flow)),
    )


@partial(jax.jit, static_argnums=(0,))
def rk_integrator(hamiltonian: Callable, state0, times):
    """
    state = exp(dt * dstate/dt) * state0 + O(dt^2)
    """

    def deriv(state, _):
        return Omega(jax.jacobian(hamiltonian)(state))

    track = odeint(deriv, state0, times)
    H = jax.vmap(hamiltonian, (0,), 0)(track)
    return (track[:, 0], track[:, 1], H)
