"""Passage of particles through material

These are energy loss and straggling effect computations,
as described e.g. in https://pdg.lbl.gov/2025/reviews/rpp2024-rev-passage-particles-matter.pdf

The data values themselves are retrieved from https://pdg.lbl.gov/2025/AtomicNuclearProperties/
"""

import functools
from dataclasses import dataclass
from typing import Protocol

import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np

from beamline.jax.types import SFloat
from beamline.units import ELECTRON_MASS


@dataclass(frozen=True)
class DensityCorrection:
    """Parameters for Sternheimer density correction

    Per Sternheimer:1952jn or PDG 34.7

    Parameters
    ----------
    bg: float
        Relativistic beta * gamma of the incident particle
    """

    C: float
    x0: float
    x1: float
    a: float
    k: float
    delta0: float
    """Only non-zero for conductors"""

    def __call__(self, bg):
        x = jnp.log10(bg)
        return jnp.where(
            x < self.x0,
            self.delta0 * jnp.pow(10, 2 * (x - self.x0)),
            jnp.where(
                x < self.x1,
                2 * jnp.log(10) * x
                - self.C
                + self.a * jnp.power(abs(self.x1 - x), self.k),
                2 * jnp.log(10) * x - self.C,
            ),
        )


@dataclass(frozen=True)
class InteractionParams:
    """Parameters of a particle's stochastic interaction with material

    Energy straggling (PDG 34.2.9) and multiple Coulomb scattering (PDG 34.3)
    for one traversal segment.
    """

    xi: SFloat
    """Landau's xi (the scaling of the dimensionless Landau parameter)"""
    kappa: SFloat
    """
    This parameter characterizes when we are in the "stochastic" regime,
    i.e. thick enough that the Landau distribution is not accurate, but not yet
    in the Gaussian regime (at which point likely the constant pc assumption is violated)

    Generally the Vavilov distribution is adequate for 0.01 < kappa < 10.
    As kappa goes to zero, the Landau distribution becomes valid. As kapppa goes to inf,
    the Gaussian distribution becomes valid.

    Some references for convolution approach to sampling vavilov dist:
    Convolution approach https://doi.org/10.1016/S0168-583X(98)00803-9
    Direct Rotondi:1990bzg
    """
    mean_energy_loss: SFloat
    """Mean energy loss (Bethe-Bloch formula)"""
    mode_energy_loss: SFloat
    """Most probable energy loss"""
    theta0: SFloat
    """RMS projected (plane) scattering angle [rad] (Highland, PDG 34.16)"""
    thickness: SFloat
    """Path length of material traversed in this segment [mm]
    The two projected planes are independent and identically distributed
    (PDG 34.18); the space angle is sqrt(2) * theta0 (PDG 34.15).
    """
    # NOTE: the following are needed by the new sampler, which only recieves
    # InteractionParams. I figured the duplication was sensible since they are
    # real interaction parameters in this case, but let me know!
    radiation_lengths: SFloat
    """Segment thickness in radiation lengths, d/X0 (dimensionless)
    Used by the mixture scattering models, where the variance is proportional to
    d/X0 and additive over segments.
    """
    beta: SFloat
    """Incident particle velocity v/c"""
    momentum: SFloat
    """Incident particle momentum |p| [MeV]"""
    charge: SFloat
    """Incident particle charge [e]"""
    Z: SFloat
    """(Average) nuclear charge of the material"""


class IncidentParticle(Protocol):
    """Necessary incident particle properties for material interactions

    The implementation in beamline.jax.kinematics.ParticleState is an example of this protocol.
    """

    @property
    def mass(self) -> SFloat:
        """Particle mass-energy in MeV"""
        ...

    @property
    def charge(self) -> SFloat:
        """Particle charge in units of e"""
        ...

    def beta(self) -> SFloat:
        """Particle velocity beta = v/c"""
        ...

    def gamma(self) -> SFloat:
        """Particle Lorentz factor gamma = 1 / sqrt(1 - beta^2)"""
        ...


DEDX_CONSTANT: float = 0.307075 * u.MeV * u.cm2 / u.mol
r"""<dE/dx> coefficient from PDG

Equal to $4 \pi N_A r_e^2 m_e c^2$
where $r_e$ is the classical electron radius,
$m_e$ is the electron mass, and $N_A$ is Avogadro's number.
"""


@dataclass(frozen=True)
class Material:
    """A material that particles may pass through"""

    # TODO: abstract and have atomic / compound materials, with the latter being made up of the former

    name: str
    """Display name"""
    Z: int
    """Atomic number"""
    mass: float
    """Atomic mass [MeV/c^2/mol]"""
    density: float
    """"Material density [MeV/c^2/mm^3]"""
    mean_excitation: float
    """Mean excitation energy [MeV] (to excite an electron)"""
    plasma_energy: float
    """Plasma energy [MeV]"""
    radiation_length: float
    """Radiation Length [MeV/c^2/mm^2]"""
    is_atomic: bool
    """True if this is an atomic element (rather than a compound)"""
    density_correction: DensityCorrection

    @staticmethod
    def _wmax(beta: SFloat, gamma: SFloat, mass: SFloat) -> SFloat:
        """Maximum energy transfer in a single collision with an electron [MeV]"""
        mass_ratio = ELECTRON_MASS * u.c_light_sq / mass
        return (
            2
            * ELECTRON_MASS
            * u.c_light_sq
            * (beta * gamma) ** 2
            / (1 + 2 * gamma * mass_ratio + mass_ratio**2)
        )

    def _mean_dedx(
        self, beta: SFloat, gamma: SFloat, mass: SFloat, charge: SFloat
    ) -> SFloat:
        """Bethe-Bloch mean stopping power <-dE/dx> [MeV/mm] (PDG 34.5)"""
        Wmax = self._wmax(beta, gamma, mass)
        prefactor = DEDX_CONSTANT * (self.Z / self.mass) * charge**2 * self.density
        return (
            0.5
            * prefactor
            / beta**2
            * (
                jnp.log(
                    2
                    * ELECTRON_MASS
                    * u.c_light_sq
                    * (beta * gamma) ** 2
                    * Wmax
                    / self.mean_excitation**2
                )
                - 2 * beta**2
                - self.density_correction(beta * gamma)
            )
        )

    def stopping_power(self, pc: SFloat, mass: SFloat, charge: SFloat) -> SFloat:
        """Mean stopping power <-dE/dx> [MeV/mm] for momentum ``pc`` [MeV]

        Args:
            pc: Momentum times c [MeV]
            mass: Incident particle mass-energy [MeV]
            charge: Incident particle charge [e]
        """
        E = jnp.hypot(mass, pc)
        return self._mean_dedx(pc / E, E / mass, mass, charge)

    def csda_range(
        self,
        pc: SFloat,
        mass: SFloat,
        charge: SFloat,
        *,
        bg_min: float = 0.05,
        order: int = 64,
    ) -> SFloat:
        """Continuous-slowing-down-approximation range [mm] at momentum ``pc`` [MeV]

        The path length over which the mean energy loss brings the particle to
        rest. Since dE = beta dp, ``R(p) = int beta / <-dE/dx> dp``, computed by
        Gauss-Legendre quadrature in log p (cf. the ODE solve in
        ``notebook/straggling_exploration.ipynb``).

        Bethe-Bloch is not valid at low velocity (and turns negative), so the
        integral is cut off at ``beta * gamma = bg_min``; the range below that
        is negligible (of order microns for muons). Compared to the PDG muon
        tables (which include low-energy corrections), muons in LiH agree to
        within 1% above 50 MeV/c, and the range is ~10% long at 25 MeV/c.

        Args:
            pc: Momentum times c [MeV] (any shape)
            mass: Incident particle mass-energy [MeV]
            charge: Incident particle charge [e]
            bg_min: Lower cutoff of the integral in beta * gamma
            order: Number of quadrature nodes
        """
        nodes, weights = np.polynomial.legendre.leggauss(order)
        pc = jnp.asarray(pc)[..., None]
        log_lo = jnp.log(bg_min * mass)
        log_hi = jnp.maximum(jnp.log(pc), log_lo)
        half = 0.5 * (log_hi - log_lo)
        p = jnp.exp(log_lo + half * (nodes + 1.0))
        beta = p / jnp.hypot(mass, p)
        # dp = p dlog(p)
        integrand = beta * p / self.stopping_power(p, mass, charge)
        return jnp.sum(half * weights * integrand, axis=-1)

    def stopping_momentum(self, min_range: float, mass: float, charge: float) -> float:
        """Momentum [MeV] below which the CSDA range is less than ``min_range`` [mm]

        A particle below this momentum will range out within ``min_range`` and
        is considered stopped. The value is computed once per arguments (by
        bisection on ``csda_range``) and cached, so this must be called with
        concrete (static) arguments; it is safe to call while tracing.
        """
        return _stopping_momentum(self, min_range, mass, charge)

    def interaction_params(
        self,
        particle: IncidentParticle,
        thickness: SFloat,
        reference_thickness: SFloat | None = None,
    ) -> InteractionParams:
        """Compute interaction parameters for a given particle and thickness"""

        beta, gamma = particle.beta(), particle.gamma()
        Wmax = self._wmax(beta, gamma, particle.mass)

        prefactor = (
            DEDX_CONSTANT * (self.Z / self.mass) * particle.charge**2 * self.density
        )

        xi = 0.5 * prefactor * thickness / beta**2
        kappa = xi / Wmax
        mean_energy_loss = (
            self._mean_dedx(beta, gamma, particle.mass, particle.charge) * thickness
        )
        # (0.2 is with density correction, 0.37 is without, per Bichsel:1998if)
        mode_energy_loss = mean_energy_loss + xi * (
            beta**2 + jnp.log(kappa) + 0.20005183774398613
        )
        # protocol-safe momentum: p = beta * gamma * m
        momentum = beta * gamma * particle.mass
        z = particle.charge
        x_ref = (
            (thickness if reference_thickness is None else reference_thickness)
            * self.density
            / self.radiation_length
        )
        x_seg = thickness * self.density / self.radiation_length
        theta0_ref = (
            (13.6 * u.MeV / (beta * momentum))
            * z
            * jnp.sqrt(x_ref)
            * (1.0 + 0.038 * jnp.log(x_ref * z**2 / beta**2))
        )
        theta0 = theta0_ref * jnp.sqrt(x_seg / x_ref)
        return InteractionParams(
            xi=xi,
            kappa=kappa,
            mean_energy_loss=mean_energy_loss,
            mode_energy_loss=mode_energy_loss,
            theta0=theta0,
            thickness=thickness,
            radiation_lengths=x_seg,
            beta=beta,
            momentum=momentum,
            charge=z,
            Z=float(self.Z),
        )


@functools.cache
def _stopping_momentum(
    material: Material, min_range: float, mass: float, charge: float
) -> float:
    """Cached implementation of ``Material.stopping_momentum``"""
    with jax.ensure_compile_time_eval():
        lo, hi = jnp.log(0.05 * mass), jnp.log(1e6 * mass)

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            short = material.csda_range(jnp.exp(mid), mass, charge) < min_range
            return jnp.where(short, mid, lo), jnp.where(short, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 60, body, (lo, hi))
        return float(jnp.exp(0.5 * (lo + hi)))


# TODO: build these programmatically from https://pdg.lbl.gov/2025/AtomicNuclearProperties/expert.html

MATERIALS: dict[str, Material] = {
    # https://pdg.lbl.gov/2025/AtomicNuclearProperties/HTML/aluminum_Al.html
    "aluminum_Al": Material(
        name="Aluminum",
        Z=13,
        mass=26.9815385 * u.g / u.mol,
        density=2.699 * u.g / u.cm3,
        mean_excitation=166.0 * u.eV,
        plasma_energy=32.86 * u.eV,
        radiation_length=24.01 * u.g / u.cm2,
        is_atomic=True,
        density_correction=DensityCorrection(
            C=4.2395, x0=0.1708, x1=3.0127, a=0.0802, k=3.6345, delta0=0.12
        ),
    ),
    # https://pdg.lbl.gov/2025/AtomicNuclearProperties/HTML/silicon_Si.html
    "silicon_Si": Material(
        name="Silicon",
        Z=14,
        mass=28.0855 * u.g / u.mol,
        density=2.329 * u.g / u.cm3,
        mean_excitation=173.0 * u.eV,
        plasma_energy=31.05 * u.eV,
        radiation_length=21.82 * u.g / u.cm2,
        is_atomic=True,
        density_correction=DensityCorrection(
            C=4.4355, x0=0.2015, x1=2.8716, a=0.1492, k=3.2546, delta0=0.14
        ),
    ),
    # https://pdg.lbl.gov/2025/AtomicNuclearProperties/HTML/lithium_hydride_LiH.html
    "lithium_hydride_LiH": Material(
        name="Lithium Hydride",
        Z=2,
        mass=(2 / 0.50321) * u.g / u.mol,
        density=0.8200 * u.g / u.cm3,
        mean_excitation=36.5 * u.eV,
        plasma_energy=18.51 * u.eV,
        radiation_length=79.61 * u.g / u.cm2,
        is_atomic=False,
        density_correction=DensityCorrection(
            C=2.3580, x0=-0.0988, x1=1.4515, a=0.9057, k=2.5849, delta0=0.0
        ),
    ),
    # MICE 6Li-enriched LiH absorber. Composition and thickness from the
    # benchmark note Table 1; radiation length calculated in arXiv:2209.10251.
    # <Z/A> = 0.56624 from Li6 0.814 / Li7 0.043 / H 0.143 by mass.
    # plasma = 28.816 sqrt(rho <Z/A>) = 18.012 eV.
    # C = 2 ln(I / plasma) + 1 = 2.4126. I is set by the electronic structure,
    # identical for Li6 and Li7, so the PDG LiH value carries over.
    # Sternheimer parameters are reused from PDG LiH.
    # TODO: Revisit Sternheimer corrections & other parameters at some point
    "lithium_hydride_MICE": Material(
        name="Lithium Hydride (MICE, 6Li-enriched)",
        Z=2,
        mass=(2 / 0.56624) * u.g / u.mol,
        density=0.69 * u.g / u.cm3,
        mean_excitation=36.5 * u.eV,
        plasma_energy=18.012 * u.eV,
        radiation_length=70.38 * u.g / u.cm2,
        is_atomic=False,
        density_correction=DensityCorrection(
            C=2.4126, x0=-0.0988, x1=1.4515, a=0.9057, k=2.5849, delta0=0.0
        ),
    ),
    # https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf
    "liquid_hydrogen_H2": Material(
        name="Liquid Hydrogen",
        Z=1,
        mass=1.008 * u.g / u.mol,
        density=0.07080 * u.g / u.cm3,
        mean_excitation=21.8 * u.eV,
        plasma_energy=7.64 * u.eV,
        radiation_length=63.04 * u.g / u.cm2,
        is_atomic=True,
        density_correction=DensityCorrection(
            C=3.0977, x0=0.4400, x1=1.8856, a=0.13483, k=5.6249, delta0=0.0
        ),
    ),
    # https://pdg.lbl.gov/2025/AtomicNuclearProperties/HTML/silicon_dioxide_fused_quartz.html
    "silicon_dioxide_SiO2": Material(
        name="Silicon Dioxide",
        Z=10,
        mass=(10 / 0.49930) * u.g / u.mol,
        density=2.200 * u.g / u.cm3,
        mean_excitation=139.2 * u.eV,
        plasma_energy=30.20 * u.eV,
        radiation_length=27.05 * u.g / u.cm2,
        is_atomic=False,
        density_correction=DensityCorrection(
            C=4.0560, x0=0.1500, x1=3.0140, a=0.08408, k=3.5064, delta0=0.0
        ),
    ),
}
