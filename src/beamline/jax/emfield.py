from __future__ import annotations

from abc import abstractmethod

import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.geometry import Volume
from beamline.jax.types import SBool, SFloat


class EMTensorField(Volume):
    """Electromagnetic field interface

    All sources of EM fields should implement this interface, which is designed to be
    compatible with the Lorentz force equation.

    Concrete implementations should implement field_strength, which returns the electric and
    magnetic field components at a given point. They should also implement the Volume interface,
    to define the region of space where the field is non-zero.
    """

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

    def __add__(self, other: EMTensorField) -> SumField:
        return SumField([self, other])


class SimpleEMField(EMTensorField):
    """Simple uniform electromagnetic field in all space"""

    E0: Cartesian3
    B0: Cartesian3

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        return jnp.inf

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

    def contains(self, point: Cartesian3) -> SBool:
        # TODO: any reason to implement this? If so, probably should set up a bounded volume hierarchy
        # return jnp.any(jnp.array([comp.contains(point) for comp in self.components]))
        return jnp.array(True)

    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        ds = jnp.array([comp.signed_distance(ray) for comp in self.components])
        return ds[jnp.argmin(jnp.abs(ds))]

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


class TransformEMField(EMTensorField):
    """A container to transform EM tensor fields

    TODO: this overrides EMTensorField.__call__, better would be to
    break EMTensorField into two parts, one for contracting, and
    another for implementing E and B fields.
    """

    transform: Transform
    field: EMTensorField
    """The field in local coordinates (i.e. before transformation)"""

    def contains(self, point: Cartesian3) -> SBool:
        return self.field.contains(self.transform.to_local(point))

    def signed_distance(self, ray: Tangent[Cartesian3]) -> SFloat:
        return self.field.signed_distance(self.transform.tangent_to_local(ray))

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        raise RuntimeError("This method should not be used, use __call__ instead")

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        in_local = self.transform.tangent_to_local(vec)
        out_local = self.field(in_local)
        return self.transform.tangent_to_global(out_local)
