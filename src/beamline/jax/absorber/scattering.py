"""Multiple Coulomb scattering for thin absorbers (PDG RPP §34.3).

Implements the Highland approximation: two independent Gaussian
deflection angles theta_x, theta_y with RMS theta_0, plus their
correlated lateral offsets y_x, y_y per PDG eqs. 34.16-17.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from beamline.jax.coordinates import Cartesian4, Tangent


def sample_scattering(state, key, material, thickness):
    """Apply multiple scattering to a particle traversing `thickness`
    of `material`. Returns (new_state, new_key).

    The state's direction is rotated by small Gaussian angles in two
    orthogonal planes. The transverse position is offset by the
    correlated lateral displacement.

    The thin-scatterer approximation assumes the particle's direction
    is nearly along z; for steeply-incident tracks you'd want a more
    careful treatment.
    """
    theta0 = material.highland_theta0(state, thickness)

    # PDG eq. 34.16: in each plane, the angle and lateral offset are
    # jointly Gaussian with correlation rho = sqrt(3)/2.
    # Sample as: z1, z2 ~ N(0,1), then
    #   y_plane = z1 * x * theta0 / sqrt(12) + z2 * x * theta0 / 2
    #   theta_plane = z2 * theta0
    # (This is the standard decomposition; see PDG eq. 34.17.)
    k1, k2, k3, k4, key_out = jax.random.split(key, 5)
    z1x = jax.random.normal(k1)
    z2x = jax.random.normal(k2)
    z1y = jax.random.normal(k3)
    z2y = jax.random.normal(k4)

    inv_sqrt12 = 1.0 / jnp.sqrt(12.0)
    y_x     = thickness * theta0 * (z1x * inv_sqrt12 + z2x * 0.5)
    theta_x = theta0 * z2x
    y_y     = thickness * theta0 * (z1y * inv_sqrt12 + z2y * 0.5)
    theta_y = theta0 * z2y

    new_state = _rotate_direction_and_offset(state, theta_x, theta_y, y_x, y_y)
    return new_state, key_out


def _rotate_direction_and_offset(state, theta_x, theta_y, y_x, y_y):
    """Update the kinematic state: rotate the momentum direction by
    small angles theta_x, theta_y, and translate the position by the
    transverse offsets y_x, y_y.

    Small-angle (linearized) version: for a particle moving primarily
    along +z, a rotation of theta_x in the x-z plane sends
       (px, py, pz) -> (pz*sin + px*cos, py, pz*cos - px*sin)
    which for small theta and |px|, |py| << pz reduces to
       (px + pz*theta_x, py + pz*theta_y, pz)
    with |p| preserved to O(theta^2). We then rescale to preserve |p|
    exactly, which matches PDG's convention that MCS is elastic.
    """
    px, py, pz = state.kin.t.x, state.kin.t.y, state.kin.t.z
    E = state.kin.t.ct
    p_old = jnp.sqrt(px**2 + py**2 + pz**2)

    # small-angle rotation of the momentum direction
    px_new = px + pz * theta_x
    py_new = py + pz * theta_y
    pz_new = pz  # second-order in (theta_x, theta_y); corrected by rescale below

    # rescale to preserve |p| exactly (MCS is elastic — direction
    # changes, magnitude does not)
    p_tmp = jnp.sqrt(px_new**2 + py_new**2 + pz_new**2)
    scale = jnp.where(p_tmp > 0, p_old / p_tmp, 1.0)
    px_new = px_new * scale
    py_new = py_new * scale
    pz_new = pz_new * scale

    new_momentum = Cartesian4.make(x=px_new, y=py_new, z=pz_new, ct=E)

    # transverse position offset (only the x and y components change;
    # z and t are untouched in the thin-scatterer approximation)
    x_pos, y_pos = state.kin.p.x, state.kin.p.y
    new_position = Cartesian4.make(
        x=x_pos + y_x,
        y=y_pos + y_y,
        z=state.kin.p.z,
        ct=state.kin.p.ct,
    )

    new_kin = Tangent(p=new_position, t=new_momentum)
    return eqx.tree_at(lambda s: s.kin, state, new_kin)
