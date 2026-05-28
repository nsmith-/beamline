"""
Absorber beamline element (stochastic, split-step)
Applies a discrete momentum kick when a particle crosses it, sampled from the Landau distribution.
Usage:

    absorber = CylindricalAbsorber(
        material=MATERIALS["silicon_dioxide_SiO2"],
        radius=100 * u.mm,
        length=10 * u.mm,
    )

    After propagating through EM fields to the absorber face:
    state, key = absorber.apply(state, key)
    Continue EM propagation...
"""

from __future__ import annotations

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp

from beamline.jax.absorber.material import Material
from beamline.jax.absorber.straggling import sample_energy_loss
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.geometry import Volume, line_cylinder_intersection, line_plane_intersection
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat

# Absorber element

class CylindricalAbsorber(Volume):
    """
    A cylindrical absorber that applies stochastic energy loss
    The absorber is centered at the origin, extending from -length/2 to
    +length/2 along the z axis.
    Energy loss is sampled from the Landau distribution. This is the
    correct limit for thin absorbers (κ < 0.01).

    Parameters
    ----------
    material : Material
        Absorber material (e.g. ``MATERIALS["silicon_dioxide_SiO2"]``)
    radius : SFloat
        Cylinder radius [mm]
    length : SFloat
        Cylinder full length [mm] (= absorber thickness for dE/dx)
    """

    material: Material
    radius: SFloat
    length: SFloat

    # -- Volume --

    def contains(self, point: Cartesian3) -> SBool:
        pcyl = point.to_cylindric()
        return (
            (pcyl.z >= -self.length / 2)
            & (pcyl.z <= self.length / 2)
            & (pcyl.rho <= self.radius)
        )

    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        tcyl, h = line_cylinder_intersection(
            ray, Cartesian3.make(), Cartesian3.make(z=self.radius)
        )
        tcyl = jnp.where(abs(h) <= self.length / 2, tcyl, jnp.inf)
        t1, uc, vc = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=-self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=-self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=-self.length / 2),
        )
        t1 = jnp.where(uc**2 + vc**2 <= self.radius**2, t1, jnp.inf)
        t2, uc, vc = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=self.length / 2),
        )
        t2 = jnp.where(uc**2 + vc**2 <= self.radius**2, t2, jnp.inf)
        ts = jnp.array([tcyl, t1, t2])
        return ts[jnp.argmin(jnp.abs(ts))]

    # -- Energy loss --

    def energy_loss(self, state: ParticleState, key: jax.Array) -> SFloat:
        """
        Sample stochastic energy loss [MeV] from the Landau distribution
        Args:
            state: Current particle state
            key: JAX PRNG key
        Returns:
            ΔE [MeV] (positive = energy lost)
        """
        params = self.material.straggling_params(state, self.length)
        loss = sample_energy_loss(key, params)

        # Can't lose more energy than kinetic energy
        kinetic_energy = state.kin.t.ct - state.mass
        return jnp.clip(loss, 0.0, 0.99 * kinetic_energy)

    def mean_energy_loss(self, state: ParticleState) -> SFloat:
        """
        Deterministic Bethe-Bloch mean energy loss (no fluctuations)
        """
        params = self.material.straggling_params(state, self.length)
        return params.mean_energy_loss

    def apply[T: ParticleState](
        self, state: T, key: jax.Array
    ) -> tuple[T, jax.Array]:
        """
        Apply stochastic energy loss to a particle crossing the absorber

        Reduces the particle's momentum magnitude along its direction of
        travel. Position is unchanged (thin-absorber approximation).

        Args:
            state: Particle state at the absorber
            key: JAX PRNG key

        Returns:
            (new_state, new_key)
        """
        key, subkey = jax.random.split(key)
        dE = self.energy_loss(state, subkey)

        # Current kinematics
        E_old = state.kin.t.ct
        px, py, pz = state.kin.t.x, state.kin.t.y, state.kin.t.z
        p_old = jnp.sqrt(px**2 + py**2 + pz**2)

        # New energy
        E_new = E_old - dE

        # New |p| from E² = p² + m²
        mass = state.mass
        p_new = jnp.sqrt(jnp.maximum(E_new**2 - mass**2, 0.0))

        # Scale momentum uniformly (preserves direction)
        scale = jnp.where(p_old > 0, p_new / p_old, 0.0)

        new_momentum = Cartesian4.make(
            x=px * scale,
            y=py * scale,
            z=pz * scale,
            ct=E_new,
        )
        new_kin = Tangent(p=state.kin.p, t=new_momentum)
        new_state = eqx.tree_at(lambda s: s.kin, state, new_kin)

        return new_state, key
