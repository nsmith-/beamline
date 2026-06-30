"""Absorber beamline element (stochastic, split-step)

An absorber is a volume of material that causes stochastic energy loss in
passing particles. Unlike EM elements (solenoids, RF cavities), the absorber
is NOT part of the continuous ODE. Instead, it applies a discrete momentum
kick when a particle crosses it, sampled from the Landau distribution
using the differentiable sampler from diff_random.

Usage::

    absorber = CylindricalAbsorber(
        material=MATERIALS["silicon_dioxide_SiO2"],
        radius=100 * u.mm,
        length=10 * u.mm,
    )

    # After propagating through EM fields to the absorber face:
    state, key = absorber.apply(state, key)
    # Continue EM propagation...

TODO: reconcile absorber propagation with EM propagation in diffrax_solve
TODO: Vavilov sampling via VavilovFast for intermediate κ
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from beamline.jax.absorber.material import Material
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.geometry import Volume, line_cylinder_intersection, line_plane_intersection
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat
from diff_random.distributions._landau import Landau_SG
from beamline.jax.absorber.scattering import sample_scattering


class CylindricalAbsorber(Volume):
    """A cylindrical absorber that applies stochastic energy loss

    The absorber is centered at the origin, extending from -length/2 to
    +length/2 along the z axis.

    This is NOT an EMTensorField — it does not participate in the ODE.
    Call :meth:`apply` to apply a discrete energy loss kick after
    propagating to the absorber location.

    Energy loss is sampled from the Landau distribution using the
    differentiable Chambers-Mallows-Stuck sampler from diff_random.

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

    # -- Volume interface --

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
        """Sample stochastic energy loss [MeV] from the Landau distribution

        Uses the differentiable Chambers-Mallows-Stuck sampler from
        diff_random. The loc/scale mapping from straggling parameters
        follows the scipy convention (see scipy.stats.landau tutorial).

        Args:
            state: Current particle state
            key: JAX PRNG key

        Returns:
            dE [MeV] (positive = energy lost)
        """
        params = self.material.straggling_params(state, self.length)

        # Map (mode_energy_loss, xi) → Landau(loc, scale)
        # and Prasanth's standard Landau normalization
        _LANDAU_STD_PEAK = -0.42931453  # Empirical peak of the standardized sampler;
                              # verified against scipy.stats.landau convention.

        dist = Landau_SG(
            loc=params.mode_energy_loss - _LANDAU_STD_PEAK * params.xi,
            scale=params.xi,
        )
        sample, _, _ = dist._generate_one_sample(key)

        kinetic_energy = state.kin.t.ct - state.mass
        return jnp.clip(sample, 0.0, 0.99 * kinetic_energy)

    def mean_energy_loss(self, state: ParticleState) -> SFloat:
        """Deterministic Bethe-Bloch mean energy loss (no fluctuations)"""
        params = self.material.straggling_params(state, self.length)
        return params.mean_energy_loss

    def apply[T: ParticleState](self, state: T, key: jax.Array) -> tuple[T, jax.Array]:
        """Apply multiple Coulomb scattering AND stochastic energy loss to a
        particle crossing the absorber.

        Scattering is applied FIRST so that the Highland formula uses the
        incoming kinematics, where beta*p is well-defined and bounded. If
        energy loss is applied first, the rare Landau high-loss tail can
        leave a particle with near-zero remaining momentum; the Highland
        formula then sees 1/(beta*p) -> infinity and produces wildly
        unphysical angles that dominate the empirical RMS even at
        sub-percent frequency. Swapping the order eliminates this
        pathology at negligible physics cost (the energy loss in a 10 mm
        absorber changes p by ~3%, shifting theta_0 by similar size --
        far smaller than Highland's own ~11% inherent accuracy).

        Reduces the particle's momentum magnitude along its direction of
        travel (Landau energy loss, PDG section 34.2), and rotates the
        direction by Highland-formula scattering angles plus a correlated
        transverse position offset (PDG section 34.3).

        The longitudinal position is unchanged (thin-absorber
        approximation).

        Args:
            state: Particle state at the absorber
            key: JAX PRNG key

        Returns:
            (new_state, new_key)
        """
        # Multiple Coulomb scattering using the incoming (pre-energy-loss)
        # state — see the docstring above for why this order matters.
        state, key = sample_scattering(state, key, self.material, self.length)

        # Stochastic Landau energy loss
        key, subkey = jax.random.split(key)
        dE = self.energy_loss(state, subkey)

        E_old = state.kin.t.ct
        px, py, pz = state.kin.t.x, state.kin.t.y, state.kin.t.z
        p_old = jnp.sqrt(px**2 + py**2 + pz**2)

        E_new = E_old - dE
        mass = state.mass
        p_new = jnp.sqrt(jnp.maximum(E_new**2 - mass**2, 0.0))

        scale = jnp.where(p_old > 0, p_new / p_old, 0.0)

        new_momentum = Cartesian4.make(
            x=px * scale,
            y=py * scale,
            z=pz * scale,
            ct=E_new,
        )
        new_kin = Tangent(p=state.kin.p, t=new_momentum)
        state = eqx.tree_at(lambda s: s.kin, state, new_kin)

        return state, key
