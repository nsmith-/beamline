"""Beam moments of a batch of particle states"""

import jax.numpy as jnp
from jax import Array

from beamline.jax.kinematics import ParticleState


def beam_moments(states: ParticleState, mask: Array | None = None) -> dict[str, Array]:
    """Optical functions and emittances of a beam at a common plane

    Computed from second moments of the kinetic phase-space coordinates, following
    the usual solenoid-channel (e.g. G4beamline/ecalc9) conventions:
        eps_t = det(cov(x, px, y, py))^(1/4) / m
        beta_t = (<x^2> + <y^2>) <pz> / (2 m eps_t)
        alpha_t = -(<x px> + <y py>) / (2 m eps_t)
        l_kin = <x py - y px>
        eps_l = det(cov(ct, E))^(1/2) / m

    Args:
        states: Particle states, with the leading axis indexing particles
        mask: Optional per-particle boolean selection (e.g. surviving particles)

    Returns:
        A dict with eps_t [mm], beta_t [mm], alpha_t, l_kin [mm MeV], eps_l [mm],
        and first/second moments: pz_mean [MeV], energy_mean [MeV], ct_mean [mm],
        r_rms [mm], energy_rms [MeV], ct_rms [mm]
    """
    pos, mom = states.kin.p.coords, states.kin.t.coords
    weights = None if mask is None else mask.astype(pos.dtype)
    mass = states.mass
    x, y, ct = pos[:, 0], pos[:, 1], pos[:, 3]
    px, py, pz, energy = mom[:, 0], mom[:, 1], mom[:, 2], mom[:, 3]

    cov_t = jnp.cov(jnp.stack([x, px, y, py]), aweights=weights)
    eps_t = jnp.linalg.det(cov_t) ** 0.25 / mass
    pz_mean = jnp.average(pz, weights=weights)
    cov_l = jnp.cov(jnp.stack([ct, energy]), aweights=weights)
    return {
        "eps_t": eps_t,
        "beta_t": (cov_t[0, 0] + cov_t[2, 2]) * pz_mean / (2 * mass * eps_t),
        "alpha_t": -(cov_t[0, 1] + cov_t[2, 3]) / (2 * mass * eps_t),
        "l_kin": jnp.average(x * py - y * px, weights=weights),
        "eps_l": jnp.sqrt(jnp.linalg.det(cov_l)) / mass,
        "pz_mean": pz_mean,
        "energy_mean": jnp.average(energy, weights=weights),
        "ct_mean": jnp.average(ct, weights=weights),
        "r_rms": jnp.sqrt(jnp.average(x**2 + y**2, weights=weights)),
        "energy_rms": jnp.sqrt(cov_l[1, 1]),
        "ct_rms": jnp.sqrt(cov_l[0, 0]),
    }
