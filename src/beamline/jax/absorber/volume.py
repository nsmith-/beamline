"""Material volumes for stochastic particle-matter interactions

A ``MaterialVolume`` is the material analogue of ``EMTensorField`` (which extends
``Volume`` for electromagnetic sources): it is a region of space (so it composes
with the boundary-aware step-size control via ``signed_time_to_boundary``) that, for a
given particle traversing a given thickness, returns the *shape parameters* of
the stochastic interaction (energy straggling now, multiple scattering later).

The stochastic stepper (``beamline.jax.integrate.stochastic``) keys on these
volumes to decide where and how to apply stochastic kicks.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import hepunits as u

from beamline.jax.absorber.material import Material, StragglingParams
from beamline.jax.coordinates import Cartesian3, Tangent, Transform
from beamline.jax.geometry import (
    CylinderVolume,
    Volume,
)
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat


class MaterialVolume(Volume):
    """A region of space filled with material that particles interact with

    Concrete implementations provide the ``Volume`` interface (``contains`` /
    ``signed_time_to_boundary``) to define the spatial extent, plus the two methods
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


class TransformMaterialVolume(MaterialVolume):
    """A MaterialVolume placed in the world via a rigid-body Transform

    Follows the same pattern as ``TransformEMField``: all queries are
    transformed into the local frame.
    """

    transform: Transform
    """Rigid transform placing the inner volume in global coordinates"""
    material: MaterialVolume
    """Inner material volume defined in its own local coordinates"""

    def characteristic_length(self) -> SFloat:
        return self.material.characteristic_length()

    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        local_state = eqx.tree_at(
            lambda x: x.kin, state, self.transform.tangent_to_local(state.kin)
        )
        return self.material.interaction_params(local_state, thickness)

    def contains(self, point: Cartesian3) -> SBool:
        return self.material.contains(self.transform.to_local(point))

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        return self.material.signed_time_to_boundary(
            self.transform.tangent_to_local(ray)
        )


class AbsorberCylinder(MaterialVolume, CylinderVolume):
    """A solid cylindrical absorber, centered at the origin with axis along z

    Geometry mirrors ``beamline.jax.rfcavity.pillbox.PillboxCavity``: the
    cylindrical side plus the two end disks bound the volume.
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
