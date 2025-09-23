"""Pillbox RF cavities"""

from dataclasses import dataclass, field
from functools import partial
import numpy as np
import hepunits as u
from scipy.special import jv, jvp, jn_zeros
import vector


from beamline.units import check_units, u as ureg


assert check_units(1, 0, -1, -1, ureg.megavolt / ureg.meter)
assert check_units(1, 1, -2, -1, ureg.tesla)
assert check_units(1, 1, -1, 0, ureg.kilogram * ureg.meter / ureg.second)


def _contract(
    vec: vector.VectorObject4D,
    *,
    E: vector.VectorObject3D,
    B: vector.VectorObject3D,
):
    r"""Contract the field 2-form with a (momentum) 4-vector, and use the metric to raise the result to a vector

    Computes $\eta^{\rho\nu}F_{\mu\nu} p^{\nu}$
    where $\eta$ is the Minkowski metric with signature (+,-,-,-)
    The result is a 4-vector

    Units: (MeV, ns, mm, e)
    E: (1, 0, -1, -1)
    B: (1, 1, -2, -1)
    vec: (1, 1, -1, 0)
    out: (2, 2, -3, -1)
    """
    Etmp = (E / u.c_light).to_xyz()
    Btmp = B.to_xyz()
    return vector.VectorObject4D(
        t=Etmp.x * vec.x + Etmp.y * vec.y + Etmp.z * vec.z,
        x=Etmp.x * vec.t + Btmp.z * vec.y - Btmp.y * vec.z,
        y=Etmp.y * vec.t - Btmp.z * vec.x + Btmp.x * vec.z,
        z=Etmp.z * vec.t + Btmp.y * vec.x - Btmp.x * vec.y,
    )


def _bessel_zero(m: int, n: int) -> float:
    return jn_zeros(m, n)[-1]


@dataclass
class PillboxTMCavity:
    """Pillbox cavitiy with TM standing wave mode

    The cavity is centered at 0 with length cavity_length
    """

    length: float
    """Length of the cavity [mm]"""
    radius: float
    """Radius of the cavity [mm]"""
    E0: float
    """Peak electric field [MeV/e/mm]"""
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
        self.vmn = _bessel_zero(self.m, self.n)
        self.frequency = (
            np.hypot(self.vmn / (2 * np.pi * self.radius), self.p / (2 * self.length))
            * u.c_light
        )
        self.wavelength = u.c_light / self.frequency

    def field_strength(
        self, position: vector.VectorObject4D
    ) -> tuple[vector.VectorObject3D, vector.VectorObject3D]:
        """Field strength at a given position"""
        cosZ = np.cos(self.p * np.pi * position.z / self.length)
        sinZ = np.sin(self.p * np.pi * position.z / self.length)
        besselarg = self.vmn * position.rho / self.radius
        bessel = jv(self.m, besselarg)
        # j_m(x)/x = (j_m-1(x) - j_m+1(x))/(2m)
        besselr = (
            (jv(self.m - 1, besselarg) - jv(self.m + 1, besselarg)) / (2 * self.m)
            if self.m != 0
            else 0.0
        )
        besselp = jvp(self.m, besselarg)
        cosPhi = np.cos(self.m * position.phi + self.rotation)
        sinPhi = np.sin(self.m * position.phi + self.rotation)
        omega = self.frequency / (2 * np.pi)
        Ez = self.E0 * cosZ * bessel * cosPhi
        Er = (
            -self.E0
            * (self.p * np.pi * self.radius)
            / (self.vmn * self.length)
            * sinZ
            * besselp
            * cosPhi
        )
        Ephi = (
            self.E0
            * (self.m * self.p * np.pi * self.radius)
            / (self.vmn * self.length)
            * sinZ
            * besselr
            * sinPhi
        )
        Br = (
            self.E0
            * (self.m * omega * self.radius)
            / (u.c_light_sq * self.vmn)
            * cosZ
            * besselr
            * sinPhi
        )
        Bphi = (
            self.E0
            * (omega * self.radius)
            / (u.c_light_sq * self.vmn)
            * cosZ
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
        Evec = vector.VectorObject3D(rho=Er, phi=Ephi, z=Ez) * tRe * boundary
        Bvec = vector.VectorObject3D(rho=Br, phi=Bphi, z=Bz) * tIm * boundary
        return Evec, Bvec

    def field_tensor(self, position: vector.VectorObject4D):
        """Field tensor at a given position"""
        Evec, Bvec = self.field_strength(position)
        return partial(_contract, E=Evec, B=Bvec)


if __name__ == "__main__":
    tm010 = PillboxTMCavity(
        length=2 * u.m,
        radius=0.5 * u.m,
        E0=5 * u.megavolt / u.m,
        m=0,
        n=1,
        p=0,
        phase=0.0,
    )
    print(tm010)
    print("TM010 frequency:", tm010.frequency / u.megahertz)

    from beamline.units import from_clhep, u as ureg

    # Confirm units
    Bphi = (
        from_clhep(tm010.E0, ureg.megavolt / ureg.meter)
        * from_clhep(tm010.frequency, ureg.megahertz)
        * from_clhep(tm010.radius, ureg.meter)
        / (2 * np.pi * tm010.vmn * ureg.c**2)
    )
    print("Bphi for TM010:", Bphi.to(ureg.microtesla))
    print(Bphi * ureg.c)
