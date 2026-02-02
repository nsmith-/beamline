from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Point, TangentVector
from beamline.jax.kinematics import ParticleState


class EMTensorField(eqx.Module):
    """Electromagnetic field tensor field, represented by its components"""

    @abstractmethod
    def field_strength(
        self, point: Point[Cartesian4]
    ) -> tuple[TangentVector[Cartesian3], TangentVector[Cartesian3]]:
        """Field strength tensor components at a given point

        Args:
            point: Point in spacetime where the field is evaluated [mm]

        Returns:
            E, B: Electric and Magnetic field components at the given point
                Electric field in [MeV/e/mm] (i.e. gigavolt/m)
                Magnetic field in [MeV*ns/e/mm^2] (i.e. kilotesla)
        """

    def __call__(self, vec: TangentVector[Cartesian4]) -> TangentVector[Cartesian4]:
        """Return the field tensor contracted with a tangent vector at a point

        The result is a tangent vector at the same point. Formally this should return
        a covector, but we raise the index using the Minkowski metric.

        Args:
            vec: Tangent four-momentum (i.e. dx is scaled by mass) [MeV]

        Returns:
            Change in [MeV/ns * MeV/e]. Scale by q/mc^2 to get dpc/dtau.
        """
        E, B = self.field_strength(vec.point)
        Etmp = E.dx.coords
        Btmp = B.dx.coords * u.c_light
        p = vec.dx.coords * u.c_light
        Egy = Etmp @ p[:3]
        pxc = Etmp[0] * p[3] + Btmp[2] * p[1] - Btmp[1] * p[2]
        pyc = Etmp[1] * p[3] - Btmp[2] * p[0] + Btmp[0] * p[2]
        pzc = Etmp[2] * p[3] + Btmp[1] * p[0] - Btmp[0] * p[1]
        return TangentVector(
            point=vec.point,
            dx=Cartesian4(coords=jnp.array([pxc, pyc, pzc, Egy])),
        )

    def __add__(self, other: EMTensorField) -> EMTensorField:
        return SumField([self, other])


class SimpleEMField(EMTensorField):
    """Simple uniform electromagnetic field in all space"""

    E0: Cartesian3
    B0: Cartesian3

    def field_strength(
        self, point: Point[Cartesian4]
    ) -> tuple[TangentVector[Cartesian3], TangentVector[Cartesian3]]:
        return (
            TangentVector(point=Point(x=point.x.to_cartesian3()), dx=self.E0),
            TangentVector(point=Point(x=point.x.to_cartesian3()), dx=self.B0),
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
        self, point: Point[Cartesian4]
    ) -> tuple[TangentVector[Cartesian3], TangentVector[Cartesian3]]:
        E_total = jnp.array([0.0, 0.0, 0.0])
        B_total = jnp.array([0.0, 0.0, 0.0])
        for comp in self.components:
            E, B = comp.field_strength(point)
            E_total.at[:].add(E.dx.coords)
            B_total.at[:].add(B.dx.coords)
        return (
            TangentVector(
                point=Point(x=point.x.to_cartesian3()),
                dx=Cartesian3(coords=E_total),
            ),
            TangentVector(
                point=Point(x=point.x.to_cartesian3()),
                dx=Cartesian3(coords=B_total),
            ),
        )


def particle_interaction(
    state: ParticleState,
    field: EMTensorField,
) -> ParticleState:
    """Compute the interaction of a particle with an electromagnetic field

    Returns a differential change in the particle state due to the Lorentz force,
    with respect to frame time.

    TODO: other independent variables (proper time, path length, etc.)
        (this could go here as a parameter, a new function, or be part of state.build_tangent)
    TODO: verlet integration / symplectic integrators ?
    """
    # unitless in this convention
    dposition_dct = state.kin.dx.coords / state.kin.dx.ct
    # Unit: [MeV/mm]
    # Note: state.mass is in MeV
    dmomentum_dctau = (state.charge / state.mass / u.c_light) * field(
        state.kin
    ).dx.coords
    dt_dtau = state.gamma()
    dkin = TangentVector(
        point=Point(x=Cartesian4(dposition_dct)),
        dx=Cartesian4(coords=dmomentum_dctau / dt_dtau),
    )
    return state.build_tangent(dkin)
