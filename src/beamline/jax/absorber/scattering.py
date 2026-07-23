"""Multiple Coulomb scattering for thin absorbers (PDG §34.3).

Implements the Highland approximation: two independent Gaussian
deflection angles theta_x, theta_y with RMS theta_0, plus their
correlated lateral offsets y_x, y_y PDG eqs. 34.16-17.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SFloat


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
    theta0 = material.interaction_params(state, thickness).theta0

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

def _perp_basis(n: Cartesian3) -> tuple[Cartesian3, Cartesian3]:
    """Orthonormal (u, v) spanning the plane perpendicular to unit vector n"""
    # reference axis chosen away from n so the cross product is well-conditioned
    near_z = jnp.abs(n.z) >= 0.9
    ref = Cartesian3.make(
        x=jnp.where(near_z, 1.0, 0.0), y=0.0, z=jnp.where(near_z, 0.0, 1.0)
    )
    u = ref.cross(n)
    u = u * (1.0 / abs(u))
    return u, n.cross(u)

def _rotate_direction_and_offset[T: ParticleState](
    state: T,
    theta_x: SFloat,
    theta_y: SFloat,
    y_x: SFloat,
    y_y: SFloat,
) -> T:
    """Deflect the momentum by ``(theta_x, theta_y)`` and offset the position

    The deflection is a rigid rotation about an axis perpendicular to the
    particle's *current* direction, so it is correct at any incidence rather
    than assuming travel along +z. Because the axis is perpendicular to the
    direction, Rodrigues' formula loses its ``axis (axis . n)(1 - cos)`` term
    and reduces to ``n' = n cos(theta) + d sin(theta)``, which preserves the
    momentum magnitude exactly (multiple scattering is elastic) with no
    rescaling. The rotation is built with ``Transform.make_axis_angle``, whose
    4x4 matrix leaves the energy component untouched.

    The angles and the lateral offsets are expressed in the same perpendicular
    basis ``(u, v)``, which preserves the per-plane angle-offset correlation
    ``rho = sqrt(3)/2`` of PDG 34.22.
    """
    p3 = Cartesian3(coords=state.kin.t.coords[..., :3])
    pmag = abs(p3)
    n = p3 * (1.0 / jnp.where(pmag > 0.0, pmag, 1.0))
    u, v = _perp_basis(n)

    # Rotation axis perpendicular to both n and the transverse deflection
    # direction. It need not be normalized: make_axis_angle divides by abs(axis).
    theta = jnp.sqrt(theta_x**2 + theta_y**2)
    axis_raw = v * theta_x - u * theta_y
    # At theta == 0 the rotation is the identity for any axis; substitute u so
    # make_axis_angle's 1/abs(axis) stays finite. The stepper passes exactly
    # zero angles when no kick is applied, so this branch is taken in practice.
    axis = Cartesian3(coords=jnp.where(theta > 0.0, axis_raw.coords, u.coords))

    rotation = Transform.make_axis_angle(axis, theta, Cartesian4.make())
    new_momentum = rotation.to_global(state.kin.t)

    offset = u * y_x + v * y_y
    new_position = Cartesian4.make(
        x=state.kin.p.x + offset.x,
        y=state.kin.p.y + offset.y,
        z=state.kin.p.z + offset.z,
        ct=state.kin.p.ct,
    )

    new_kin = Tangent(p=new_position, t=new_momentum)
    return eqx.tree_at(lambda s: s.kin, state, new_kin)
