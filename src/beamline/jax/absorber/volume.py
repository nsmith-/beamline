"""Material volumes for stochastic particle-matter interactions

A ``MaterialVolume`` is the material analogue of ``EMTensorField`` (which extends
``Volume`` for electromagnetic sources): it is a region of space (so it composes
with the boundary-aware step-size control via ``signed_distance``) that, for a
given particle traversing a given thickness, returns the *shape parameters* of
the stochastic interaction (energy straggling now, multiple scattering later).

The stochastic stepper (``beamline.jax.integrate.stochastic``) keys on these
volumes to decide where and how to apply stochastic kicks.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.absorber.material import Material, StragglingParams
from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import (
    Volume,
    line_cylinder_intersection,
    line_plane_intersection,
)
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat


class MaterialVolume(Volume):
    """A region of space filled with material that particles interact with

    Concrete implementations provide the ``Volume`` interface (``contains`` /
    ``signed_distance``) to define the spatial extent, plus the two methods
    below describing the stochastic interaction.
    """

    @abstractmethod
    def characteristic_length(self) -> SFloat:
        """Maximum thickness of a single stochastic segment [mm]

        Used by the stepper to subdivide a traversal as
        ``min(characteristic_length, distance_to_boundary)``. It keeps the
        per-step ``dE/dx`` roughly constant and the straggling distribution
        well-approximated, and it controls how finely the straggling
        distribution is built up by repeated kicks.
        """

    @abstractmethod
    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        """Shape parameters of the stochastic interaction for ``state``

        Args:
            state: The incident particle state (satisfies the
                ``beamline.jax.absorber.material.IncidentParticle`` protocol).
            thickness: Path length traversed through the material [mm].

        Returns:
            ``StragglingParams`` for energy straggling. Structured so that a
            future ``MultipleScatteringParams`` can be returned alongside.
        """


class AbsorberCylinder(MaterialVolume):
    """A solid cylindrical absorber, centered at the origin with axis along z

    Geometry mirrors ``beamline.jax.rfcavity.pillbox.PillboxCavity``: the
    cylindrical side plus the two end disks bound the volume.

    TODO: factor out common shapes into geometry utilities
    """

    material: Material = eqx.field(static=True)
    """The material filling the cylinder (a static, hashable config object)"""
    radius: SFloat
    """Cylinder radius [mm]"""
    length: SFloat
    """Cylinder length along z [mm]"""
    char_length: SFloat = 5.0 * u.mm
    """Characteristic segment length for stochastic subdivision [mm]"""

    def characteristic_length(self) -> SFloat:
        return self.char_length

    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        return self.material.straggling_params(state, thickness)

    def contains(self, point: Cartesian3) -> SBool:
        pcyl = point.to_cylindric()
        return (
            (pcyl.z >= -self.length / 2)
            & (pcyl.z <= self.length / 2)
            & (pcyl.rho <= self.radius)
        )

    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        # cylindrical side
        tcyl, h = line_cylinder_intersection(
            ray, Cartesian3.make(), Cartesian3.make(z=self.radius)
        )
        tcyl = jnp.where(abs(h) <= self.length / 2, tcyl, jnp.inf)
        # end disks. line_plane_intersection returns (u, v) in units of the
        # basis vectors (here scaled by radius), so the in-disk test is the unit
        # disk u**2 + v**2 <= 1.
        t1, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=-self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=-self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=-self.length / 2),
        )
        t1 = jnp.where(uu**2 + vv**2 <= 1.0, t1, jnp.inf)
        t2, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=self.length / 2),
        )
        t2 = jnp.where(uu**2 + vv**2 <= 1.0, t2, jnp.inf)
        ts = jnp.array([tcyl, t1, t2])
        return ts[jnp.argmin(jnp.abs(ts))]
