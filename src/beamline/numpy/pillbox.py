"""Pillbox RF cavities"""

from dataclasses import dataclass, field
from typing import Literal

import hepunits as u
import numpy as np
import vector
from scipy.special import jn_zeros, jnp_zeros, jv, jvp

from beamline.numpy.bessel import jv_over_z
from beamline.numpy.emfield import FieldStrength
from beamline.numpy.emfield import EMTensorField


def _bessel_zero(m: int, n: int) -> float:
    return jn_zeros(m, n)[-1]


@dataclass
class PillboxCavity(EMTensorField):
    """Pillbox cavitiy with standing wave mode

    The cavity is centered at 0 with length cavity_length
    """

    length: float
    """Length of the cavity [mm]"""
    radius: float
    """Radius of the cavity [mm]"""
    E0: float
    """Peak electric field [MeV/e/mm]"""
    mode: Literal["TE", "TM"]
    """Resonant mode type (transverse electric or transverse magnetic)"""
    m: int
    """Azimuthal mode number"""
    n: int
    """Radial mode number"""
    p: int
    """Longitudinal mode number"""
    phase: float
    """Time phase offset [rad]"""
    rotation: float = 0.0
    """Rotation around the z axis [rad]

    Only relevant for m > 0 modes
    """
    vmn: float = field(init=False, repr=False)
    """Bessel zero"""
    frequency: float = field(init=False, repr=False)
    """Frequency of the mode [GHz]"""
    wavelength: float = field(init=False, repr=False)
    """Wavelength of the mode [mm]"""

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.mode not in ("TE", "TM"):
            raise ValueError("mode must be 'TE' or 'TM'")
        if self.mode == "TE" and self.p == 0:
            raise ValueError("p must be >= 1 for TE modes")

        if self.mode == "TM":
            self.vmn = jn_zeros(self.m, self.n)[-1]
        else:
            self.vmn = jnp_zeros(self.m, self.n)[-1]
        self.frequency = (
            np.hypot(self.vmn / (2 * np.pi * self.radius), self.p / (2 * self.length))
            * u.c_light
        )
        self.wavelength = u.c_light / self.frequency

    def field_strength(
        self, position: vector.VectorObject4D
    ):
        """Field strength at a given position"""
        TMcosZ = np.cos(self.p * np.pi * position.z / self.length)
        TMsinZ = np.sin(self.p * np.pi * position.z / self.length)
        # In TE mode, we need to swap sin and cos so that Ez=0 at the ends
        if self.mode == "TE":
            TMcosZ, TMsinZ = TMsinZ, TMcosZ
        besselarg = self.vmn * position.rho / self.radius
        bessel = jv(self.m, besselarg)
        besselr_sinPhi = (
            jv_over_z(self.m, besselarg)
            * np.sin(self.m * (position.phi + self.rotation))
            if self.m != 0
            else 0.0
        )
        besselp = jvp(self.m, besselarg)
        cosPhi = np.cos(self.m * (position.phi + self.rotation))
        omega = self.frequency * 2 * np.pi
        Ez = self.E0 * TMcosZ * bessel * cosPhi
        Er = (
            -self.E0
            * (self.p * np.pi * self.radius)
            / (self.vmn * self.length)
            * TMsinZ
            * besselp
            * cosPhi
        )
        Ephi = (
            self.E0
            * (self.m * self.p * np.pi * self.radius)
            / (self.vmn * self.length)
            * TMsinZ
            * besselr_sinPhi
        )
        Br = (
            self.E0
            * (self.m * omega * self.radius)
            / (u.c_light_sq * self.vmn)
            * TMcosZ
            * besselr_sinPhi
        )
        Bphi = (
            self.E0
            * (omega * self.radius)
            / (u.c_light_sq * self.vmn)
            * TMcosZ
            * besselp
            * cosPhi
        )
        Bz = 0.0
        tRe = np.cos(omega * position.t / u.c_light + self.phase)
        tIm = np.sin(omega * position.t / u.c_light + self.phase)
        boundary = 1.0 * (
            (position.z >= 0.0)
            & (position.z <= self.length)
            & (position.rho <= self.radius)
        )
        rhohat = vector.VectorObject2D(rho=1.0, phi=position.phi)
        phihat = vector.VectorObject2D(rho=1.0, phi=position.phi + np.pi / 2)
        Evec = (
            vector.VectorObject3D(
                x=Er * rhohat.x + Ephi * phihat.x,
                y=Er * rhohat.y + Ephi * phihat.y,
                z=Ez,
            )
            * tRe
            * boundary
        )
        Bvec = (
            vector.VectorObject3D(
                x=Br * rhohat.x + Bphi * phihat.x,
                y=Br * rhohat.y + Bphi * phihat.y,
                z=Bz,
            )
            * tIm
            * boundary
        )
        if self.mode == "TE":
            Evec, Bvec = Bvec * u.c_light, Evec / u.c_light
        return FieldStrength(Evec, Bvec)
