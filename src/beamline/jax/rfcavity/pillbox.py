"""Pillbox RF cavities"""

from typing import Literal

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
from scipy.special import jn_zeros, jnp_zeros

from beamline.jax import bessel
from beamline.jax.axisymmetric import ReducedComponents, axisymmetric_field_strength
from beamline.jax.coordinates import Cartesian3, Cartesian4, Cylindric3, Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.geometry import (
    CylinderVolume,
)
from beamline.jax.types import SFloat


class PillboxCavity(EMTensorField, CylinderVolume):
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
    mode: Literal["TE", "TM"] = eqx.field(static=True)
    """Resonant mode type (transverse electric or transverse magnetic)"""
    m: int = eqx.field(static=True)
    """Azimuthal mode number"""
    n: int = eqx.field(static=True)
    """Radial mode number"""
    p: int = eqx.field(static=True)
    """Longitudinal mode number"""
    phase: SFloat
    """Time phase offset [rad]"""
    rotation: SFloat = 0.0
    """Rotation around the z axis [rad]

    Only relevant for m > 0 modes
    """

    def __init__(
        self,
        length: SFloat,
        frequency: SFloat,
        E0: SFloat,
        mode: Literal["TE", "TM"],
        m: int,
        n: int,
        p: int,
        phase: SFloat,
        rotation: SFloat = 0.0,
    ):
        # the dataclass constructor doesn't have the correct signature because of CylinderVolume's abstract radius
        self.length = length
        self.frequency = frequency
        self.E0 = E0
        self.mode = mode
        self.m = m
        self.n = n
        self.p = p
        self.phase = phase
        self.rotation = rotation
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
        if self.m == 0:
            # axisymmetric: assemble Cartesian components directly, smooth through the axis
            def outside(point: Cartesian4):
                zero = Tangent(p=point.to_cartesian3(), t=Cartesian3.make())
                return zero, zero

            return jax.lax.cond(
                self.contains(point.to_cartesian3()),
                lambda point: axisymmetric_field_strength(
                    point, self._reduced_field_m0
                ),
                outside,
                point,
            )
        # TODO: m >= 1 modes are not axisymmetric, and still go through the
        # cylindrical basis change, which is singular on the axis
        E, B = jax.lax.cond(
            self.contains(point.to_cartesian3()),
            lambda p, ct: self._cylindric_field(p, ct),
            lambda p, ct: (
                Tangent(p=p, t=Cylindric3.make()),
                Tangent(p=p, t=Cylindric3.make()),
            ),
            point.to_cylindric3(),
            point.ct,
        )
        return E.to_cartesian(), B.to_cartesian()

    def potential(self, point: Cartesian4) -> Tangent[Cartesian4]:
        """Four-potential in the temporal gauge (phi = 0)

        The fields are E = E_s(x) cos(theta), B = B_s(x) sin(theta) with
        theta = omega t + phase, so A = -E_s(x) sin(theta) / omega gives
        E = -dA/dt and, by Faraday's law (curl E_s = -omega B_s), B = curl A.
        Since sin(theta) = cos(theta - pi/2), this is the electric field a quarter
        period earlier, divided by -omega. Inside the (charge-free) cavity
        div E_s = 0, so this is also the Coulomb gauge.

        Like the field, the potential is zero outside the cavity. Its component
        normal to the end plates (A_z, for TM modes) is nonzero there, so A jumps
        across the plates, as E does: canonical momenta p + qA jump when a
        particle enters or leaves the cavity, while the mechanical momentum does
        not.
        """
        omega = self.frequency * 2 * jnp.pi
        earlier = Cartesian4.make(
            x=point.x,
            y=point.y,
            z=point.z,
            ct=point.ct - jnp.pi / 2 * u.c_light / omega,
        )
        E, _ = self.field_strength(earlier)
        A = -E.t.coords / omega
        return Tangent(
            p=point,
            t=Cartesian4(
                coords=jnp.concatenate([A, jnp.zeros_like(A[..., :1])], axis=-1)
            ),
        )

    def _reduced_field_m0(
        self, rho2: SFloat, z: SFloat, ct: SFloat
    ) -> tuple[ReducedComponents, ReducedComponents]:
        """Reduced components (see ReducedComponents) of an m = 0 mode

        Same fields as _cylindric_field with m = 0, where J_0' = -J_1, written
        with J_0(kr) and J_1(kr) / kr as smooth functions of rho^2.
        """
        zrel = z / self.length + 0.5
        kp = self.p * jnp.pi / self.length
        coskz = jnp.cos(self.p * jnp.pi * zrel)
        sinkz = jnp.sin(self.p * jnp.pi * zrel)
        kmn = self.vmn / self.radius
        j0, j1x = bessel.j0_j1x(kmn**2 * rho2)
        # (J_m' = -J_1) / rho = -kmn * J_1(kr) / kr
        jmp_over_rho = -kmn * j1x
        omega = self.frequency * 2 * jnp.pi
        Et = self.E0 * kp / kmn
        Bt = self.E0 * omega / (u.c_light_sq * kmn)
        zero = jnp.zeros_like(j0)

        if self.mode == "TM":
            E = ReducedComponents(
                rho=-Et * jmp_over_rho * sinkz, phi=zero, z=self.E0 * j0 * coskz
            )
            B = ReducedComponents(rho=zero, phi=Bt * jmp_over_rho * coskz, z=zero)
        else:  # TE mode
            Et, Bt = Bt * u.c_light, Et / u.c_light
            B0 = self.E0 / u.c_light
            E = ReducedComponents(rho=zero, phi=Et * jmp_over_rho * sinkz, z=zero)
            B = ReducedComponents(
                rho=Bt * jmp_over_rho * coskz, phi=zero, z=B0 * j0 * sinkz
            )
        tRe = jnp.cos(omega * ct / u.c_light + self.phase)
        tIm = jnp.sin(omega * ct / u.c_light + self.phase)
        scale = lambda F, t: ReducedComponents(F.rho * t, F.phi * t, F.z * t)  # noqa: E731
        return scale(E, tRe), scale(B, tIm)

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
        E = E * tRe
        B = B * tIm
        return E, B
