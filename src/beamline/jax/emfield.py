from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.kinematics import ParticleState


class EMTensorField(eqx.Module):
    """Electromagnetic field tensor field, represented by its components"""

    @abstractmethod
    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        """Field strength tensor components at a given point

        Args:
            point: Point in spacetime where the field is evaluated [mm]

        Returns:
            E, B: Electric and Magnetic field components at the given point
                Electric field in [MeV/e/mm] (i.e. gigavolt/m)
                Magnetic field in [MeV*ns/e/mm^2] (i.e. kilotesla)
        """

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        """Return the field tensor contracted with a tangent vector at a point

        The result is a tangent vector at the same point. Formally this should return
        a covector, but we raise the index using the Minkowski metric.

        Args:
            vec: Tangent four-momentum (i.e. dx is scaled by mass) [MeV]

        Returns:
            Change in [MeV/mm * MeV/e]. Scale by q/mc^2 to get dpc/dctau.
        """
        E, B = self.field_strength(vec.p)
        Etmp = E.t
        Btmp = B.t * u.c_light
        p = vec.t
        Egy = Etmp.x * p.x + Etmp.y * p.y + Etmp.z * p.z
        pxc = Etmp.x * p.ct + Btmp.z * p.y - Btmp.y * p.z
        pyc = Etmp.y * p.ct - Btmp.z * p.x + Btmp.x * p.z
        pzc = Etmp.z * p.ct + Btmp.y * p.x - Btmp.x * p.y
        return Tangent(
            p=vec.p,
            t=Cartesian4.make(x=pxc, y=pyc, z=pzc, ct=Egy),
        )

    def __add__(self, other: EMTensorField) -> EMTensorField:
        return SumField([self, other])


class SimpleEMField(EMTensorField):
    """Simple uniform electromagnetic field in all space"""

    E0: Cartesian3
    B0: Cartesian3

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        return (
            Tangent(p=point.to_cartesian3(), t=self.E0),
            Tangent(p=point.to_cartesian3(), t=self.B0),
        )


class SumField(EMTensorField):
    components: list[EMTensorField]

    def __init__(self, components: list[EMTensorField]):
        self.components = []
        for comp in components:
            if isinstance(comp, SumField):
                self.components.extend(comp.components)
            else:
                self.components.append(comp)

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        E_total = jnp.array([0.0, 0.0, 0.0])
        B_total = jnp.array([0.0, 0.0, 0.0])
        for comp in self.components:
            E, B = comp.field_strength(point)
            E_total.at[:].add(E.t.coords)
            B_total.at[:].add(B.t.coords)
        return (
            Tangent(
                p=point.to_cartesian3(),
                t=Cartesian3(coords=E_total),
            ),
            Tangent(
                p=point.to_cartesian3(),
                t=Cartesian3(coords=B_total),
            ),
        )


class TransformEMField(eqx.Module):
    """A container to transform EM tensor fields (i.e. (1,1) tensors)"""

    transform: Transform
    field: Callable[[Tangent[Cartesian4]], Tangent[Cartesian4]]
    """The field in local coordinates (i.e. before transformation)"""

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        in_local = self.transform.tangent_to_local(vec)
        out_local = self.field(in_local)
        return self.transform.tangent_to_global(out_local)


def particle_interaction[T: ParticleState](state: T, field: EMTensorField) -> T:
    """Compute the interaction of a particle with an electromagnetic field

    Returns a differential change in the particle state due to the Lorentz force,
    with respect to frame time.

    TODO: other independent variables (proper time, path length, etc.)
        (this could go here as a parameter, a new function, or be part of state.build_tangent)
    TODO: verlet integration / symplectic integrators ?
    """
    # Note: to have ctau be the independent variable, divide by mc^2 instead of E (kin.t.ct)
    # unitless in this convention
    dposition_dct = state.kin * (1 / state.kin.t.ct)
    # Unit: [MeV/mm]
    dmomentum_dct = (state.charge / state.kin.t.ct) * field(state.kin)
    dkin = Tangent(
        p=dposition_dct.t,
        t=dmomentum_dct.t,
    )
    return state.build_tangent(dkin)
