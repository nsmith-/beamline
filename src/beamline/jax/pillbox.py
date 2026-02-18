"""Pillbox RF cavities"""

from typing import Literal

import hepunits as u
import jax
import jax.numpy as jnp
from scipy.special import jn_zeros, jnp_zeros

from beamline.jax import bessel
from beamline.jax.coordinates import Cartesian3, Cartesian4, Cylindric3, Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.types import SFloat


class PillboxCavity(EMTensorField):
    """Pillbox cavitiy with standing wave mode

    The cavity is centered at the origin, with the z axis along the cavity axis.
    The phase is defined such that at t=0 and z=0, the electric field on axis is
    E0 * cos(phase) for a TM010 cavity.
    """

    length: SFloat
    """Length of the cavity [mm]"""
    frequency: SFloat
    """Resonant frequency of the mode [GHz]"""
    E0: SFloat
    """Peak electric field [MeV/e/mm]"""
    mode: Literal["TE", "TM"]
    """Resonant mode type (transverse electric or transverse magnetic)"""
    m: int
    """Azimuthal mode number"""
    n: int
    """Radial mode number"""
    p: int
    """Longitudinal mode number"""
    phase: SFloat
    """Time phase offset [rad]"""
    rotation: SFloat = 0.0
    """Rotation around the z axis [rad]

    Only relevant for m > 0 modes
    """

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.mode not in ("TE", "TM"):
            raise ValueError("mode must be 'TE' or 'TM'")
        if self.mode == "TE" and self.p == 0:
            raise ValueError("p must be >= 1 for TE modes")
        if self.frequency / u.c_light < self.p / (2 * self.length):
            raise ValueError("Frequency is too low for the given length and p")

    @property
    def vmn(self) -> SFloat:
        """Bessel zero for the given mode numbers"""
        if self.mode == "TM":
            return jn_zeros(self.m, self.n)[-1]
        else:
            return jnp_zeros(self.m, self.n)[-1]

    @property
    def radius(self) -> SFloat:
        freq2 = (self.frequency / u.c_light) ** 2
        long2 = (self.p / 2 / self.length) ** 2
        return self.vmn / (2 * jnp.pi * jnp.sqrt(freq2 - long2))

    @property
    def wavelength(self) -> SFloat:
        return u.c_light / self.frequency

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        """Field strength at a given position"""
        pcyl = point.to_cylindric3()
        E, B = self._cylindric_field(pcyl, point.ct)
        return E.to_cartesian(), B.to_cartesian()

    def _cylindric_field(
        self, pcyl: Cylindric3, ct: SFloat
    ) -> tuple[Tangent[Cylindric3], Tangent[Cylindric3]]:
        # Usual formulas are for z=0 to L, we shift it to -L/2 to L/2
        zrel = pcyl.z / self.length + 0.5
        kp = self.p * jnp.pi / self.length
        coskz = jnp.cos(self.p * jnp.pi * zrel)
        sinkz = jnp.sin(self.p * jnp.pi * zrel)
        kmn = self.vmn / self.radius
        kr = kmn * pcyl.rho
        jm, jmp = jax.value_and_grad(bessel.jv, argnums=1)(self.m, kr)
        kr0 = jax.lax.select(kr != 0.0, kr, 1e-6)  # avoid NaN
        jmr = jax.lax.select(
            kr != 0.0,
            self.m * jm / kr0,
            0.5 if self.m == 1 else 0.0,
        )
        sinPhi = jnp.sin(self.m * (pcyl.phi + self.rotation))
        cosPhi = jnp.cos(self.m * (pcyl.phi + self.rotation))
        omega = self.frequency * 2 * jnp.pi
        Et = self.E0 * kp / kmn
        Bt = self.E0 * omega / (u.c_light_sq * kmn)
        E0 = self.E0

        if self.mode == "TM":
            Erho = -Et * jmp * cosPhi * sinkz  # m/r j
            Ephi = Et * jmr * sinPhi * sinkz  # m^2 /kr^2
            Ez = E0 * jm * cosPhi * coskz
            Brho = Bt * jmr * sinPhi * coskz
            Bphi = Bt * jmp * cosPhi * coskz
            Bz = jnp.zeros_like(Ez)
            E = Tangent(p=pcyl, t=Cylindric3.make(rho=Erho, phi=Ephi, z=Ez))
            B = Tangent(p=pcyl, t=Cylindric3.make(rho=Brho, phi=Bphi, z=Bz))
        else:  # TE mode
            Et, Bt = Bt * u.c_light, Et / u.c_light
            B0 = self.E0 / u.c_light
            Erho = Et * jmr * sinPhi * sinkz
            Ephi = Et * jmp * cosPhi * sinkz
            Ez = jnp.zeros_like(Erho)
            Brho = Bt * jmp * cosPhi * coskz
            Bphi = -Bt * jmr * sinPhi * coskz
            Bz = B0 * jm * cosPhi * sinkz
            E = Tangent(p=pcyl, t=Cylindric3.make(rho=Erho, phi=Ephi, z=Ez))
            B = Tangent(p=pcyl, t=Cylindric3.make(rho=Brho, phi=Bphi, z=Bz))
        tRe = jnp.cos(omega * ct / u.c_light + self.phase)
        tIm = jnp.sin(omega * ct / u.c_light + self.phase)
        boundary = 1.0 * ((zrel >= 0.0) & (zrel <= 1.0) & (pcyl.rho <= self.radius))
        E = E * tRe * boundary
        B = B * tIm * boundary
        return E, B
