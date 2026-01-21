import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import hepunits as u
import numpy as np
import vector

pytree = vector.register_pytree()


@dataclass
class FieldStrength:
    r"""Electromagnetic field strength 2-form

    i.e. the antisymmetric tensor $F_{\mu\nu}$"""

    E: vector.VectorObject3D
    "Electric field in [MeV/e/mm]"
    B: vector.VectorObject3D
    "Magnetic field in [MeV*ns/e/mm^2]"

    def tree_flatten(self):
        return ((self.E, self.B), ())

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple,
        children: tuple[vector.VectorObject3D, vector.VectorObject3D],
    ):
        E, B = children
        return cls(E, B)

    def contract(self, p: vector.MomentumObject4D) -> vector.MomentumObject4D:
        r"""Contract the field 2-form with a momentum vector, and use the metric to raise the result to a vector

        Computes $\eta^{\rho\nu}F_{\mu\nu} p^{\nu}$
        where $\eta$ is the Minkowski metric with signature (+,-,-,-)

        `p` should be in units of MeV/c, i.e. (E/c, px, py, pz)
        The result is a 4-vector in units of [MeV^2 * ns^2 / e / mm^3]
        """
        # Pre-convert to cartesian for performance
        Etmp = (self.E / u.c_light).to_xyz()
        Btmp = self.B.to_xyz()
        return vector.MomentumObject4D(
            t=Etmp.x * p.x + Etmp.y * p.y + Etmp.z * p.z,
            px=Etmp.x * p.t + Btmp.z * p.y - Btmp.y * p.z,
            py=Etmp.y * p.t - Btmp.z * p.x + Btmp.x * p.z,
            pz=Etmp.z * p.t + Btmp.y * p.x - Btmp.x * p.y,
        )


pytree.register_node_class()(FieldStrength)  # type: ignore[arg-type]


class EMTensorField(ABC):
    @abstractmethod
    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        """Evaluate the field tensor at a given position"""
        ...

    def __add__(self, other: "EMTensorField") -> "SumField":
        return SumField([self, other])


class SumField(EMTensorField):
    components: list[EMTensorField]

    def __init__(self, components: list[EMTensorField]):
        self.components = []
        for comp in components:
            if isinstance(comp, SumField):
                self.components.extend(comp.components)
            else:
                self.components.append(comp)

    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        E = vector.VectorObject3D(x=0.0, y=0.0, z=0.0)
        B = vector.VectorObject3D(x=0.0, y=0.0, z=0.0)
        for comp in self.components:
            F = comp.field_strength(position)
            E += F.E
            B += F.B
        return FieldStrength(E=E, B=B)


class StaticUniformField(EMTensorField):
    E: vector.VectorObject3D
    B: vector.VectorObject3D

    def __init__(self, E: vector.VectorObject3D, B: vector.VectorObject3D):
        self.E = E
        self.B = B

    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        return FieldStrength(E=self.E, B=self.B)


@dataclass
class TiledField(EMTensorField):
    """Tile a field in z periodically

    The field is not stacked, so one should ensure that the spacing is sufficient
    that the fields do not overlap too much.
    """

    field: EMTensorField
    "Field of one object"
    z_spacing: float
    "Spacing in z [mm]"
    ct_advance: float | None
    "Advance in ct for the each z tile [mm] (if none, no tiling in t)"

    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        zmod = float(position.z % self.z_spacing)
        if self.ct_advance is None:
            ct_adv = 0.0
        else:
            ct_adv = math.floor(position.z / self.z_spacing) * self.ct_advance
        tmod = float(position.t + ct_adv)
        # TODO: the need to wrap with float() is a side effect of the vector pytree wrapper not downcasting numpy scalars
        pmod = vector.VectorObject4D(
            x=float(position.x),
            y=float(position.y),
            z=zmod,
            t=tmod,
        )
        return self.field.field_strength(pmod)


def _scanfield(field: EMTensorField, *, x: float, y: float, z: float, ct: float):
    "Helper function to be used with np.vectorize"
    pos = vector.obj(x=x, y=y, z=z, t=ct)
    res = field.field_strength(pos)
    return res.E.x, res.E.y, res.E.z, res.B.x, res.B.y, res.B.z


_QUIVER_SCALE = 20
"quiver scale factor"


def plotXY(
    *,
    field: EMTensorField,
    axE,
    axB,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    scaleE: float,
    scaleB: float | None = None,
    z: float = 0.0,
    ct: float = 0.0,
):
    """Plot the XY field at a given z and t position

    field : EMTensorField
        The electromagnetic field to plot
    axE : matplotlib.axes
        Axes to plot the electric field on
    axB : matplotlib.axes
        Axes to plot the magnetic field on
    xlim : tuple[float, float]
        x limits in [mm]
    ylim : tuple[float, float]
        y limits in [mm]
    scaleE : float
        Color scale for electric field in [MV/mm]
    scaleB : float | None
        Color scale for magnetic field. If None, uses scaleE/c
    z : float
        z position to plot at [mm]
    ct : float
        ct position to plot at [mm]
    """
    scaleE = abs(scaleE)
    scaleB = abs(scaleB) if scaleB is not None else scaleE / u.c_light

    xvals, yvals = np.meshgrid(
        np.linspace(*xlim, 50),
        np.linspace(*ylim, 50),
        indexing="ij",
    )
    Ex, Ey, Ez, Bx, By, Bz = np.vectorize(partial(_scanfield, field, z=z, ct=ct))(
        x=xvals, y=yvals
    )
    Emax = np.sqrt(Ex**2 + Ey**2 + Ez**2).max()
    Bmax = np.sqrt(Bx**2 + By**2 + Bz**2).max()

    unitL = u.m
    unitE = u.megavolt / u.m
    unitB = u.tesla

    axE.pcolormesh(
        xvals / unitL,
        yvals / unitL,
        Ez / unitE,
        cmap="bwr",
        vmin=-scaleE / unitE,
        vmax=scaleE / unitE,
    )
    axE.quiver(
        xvals / unitL,
        yvals / unitL,
        Ex / unitE,
        Ey / unitE,
        angles="xy",
        scale=_QUIVER_SCALE * scaleE / unitE,
    )
    axE.set_aspect("equal")
    axE.set_xlabel("x [m]")
    axE.set_ylabel("y [m]")
    axE.set_title(f"Electric Field (max {Emax / unitE:.1f} MV/m)")

    axB.pcolormesh(
        xvals / unitL,
        yvals / unitL,
        Bz / unitB,
        cmap="bwr",
        vmin=-scaleB / unitB,
        vmax=scaleB / unitB,
    )
    axB.quiver(
        xvals / unitL,
        yvals / unitL,
        Bx / unitB,
        By / unitB,
        angles="xy",
        scale=_QUIVER_SCALE * scaleB / unitB,
    )
    axB.set_aspect("equal")
    axB.set_xlabel("x [m]")
    axB.set_ylabel("y [m]")
    axB.set_title(f"Magnetic Field (max {Bmax / unitB:.1f} T)")


def plotZT(
    *,
    field: EMTensorField,
    axE,
    axB,
    zlim: tuple[float, float],
    ctlim: tuple[float, float],
    scaleE: float,
    scaleB: float | None = None,
    x: float = 0.0,
    y: float = 0.0,
):
    """Plot the Z-T field at a given x and y position

    field : EMTensorField
        The electromagnetic field to plot
    axE : matplotlib.axes
        Axes to plot the electric field on
    axB : matplotlib.axes
        Axes to plot the magnetic field on
    zlim : tuple[float, float]
        z limits in [mm]
    ctlim : tuple[float, float]
        ct limits in [mm]
    scaleE : float
        Color scale for electric field in [MV/mm]
    scaleB : float | None
        Color scale for magnetic field. If None, uses scaleE/c
    x : float
        x position to plot at [mm]
    y : float
        y position to plot at [mm]
    """
    scaleE = abs(scaleE)
    scaleB = abs(scaleB) if scaleB is not None else scaleE / u.c_light

    zvals, ctvals = np.meshgrid(
        np.linspace(*zlim, 50),
        np.linspace(*ctlim, 50),
        indexing="ij",
    )
    Ex, Ey, Ez, Bx, By, Bz = np.vectorize(partial(_scanfield, field, x=x, y=y))(
        z=zvals, ct=ctvals
    )
    Emax = np.sqrt(Ex**2 + Ey**2 + Ez**2).max()
    Bmax = np.sqrt(Bx**2 + By**2 + Bz**2).max()

    unitL = u.m
    unitE = u.megavolt / u.m
    unitB = u.tesla

    axE.pcolormesh(
        zvals / unitL,
        ctvals / unitL,
        Ez / unitE,
        cmap="bwr",
        vmin=-scaleE / unitE,
        vmax=scaleE / unitE,
    )
    axE.quiver(
        zvals / unitL,
        ctvals / unitL,
        Ex / unitE,
        Ey / unitE,
        angles="xy",
        scale=_QUIVER_SCALE * scaleE / unitE,
    )
    axE.set_aspect("equal")
    axE.set_xlabel("z [m]")
    axE.set_ylabel("ct [m]")
    axE.set_title(f"Electric Field (max {Emax / unitE:.1f} MV/m)")

    axB.pcolormesh(
        zvals / unitL,
        ctvals / unitL,
        Bz / unitB,
        cmap="bwr",
        vmin=-scaleB / unitB,
        vmax=scaleB / unitB,
    )
    axB.quiver(
        zvals / unitL,
        ctvals / unitL,
        Bx / unitB,
        By / unitB,
        angles="xy",
        scale=_QUIVER_SCALE * scaleB / unitB,
    )
    axB.set_aspect("equal")
    axB.set_xlabel("z [m]")
    axB.set_ylabel("ct [m]")
    axB.set_title(f"Magnetic Field (max {Bmax / unitB:.1f} T)")
