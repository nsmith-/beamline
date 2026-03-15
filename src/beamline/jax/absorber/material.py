"""Passage of particles through material

These are energy loss and straggling effect computations,
as described e.g. in https://pdg.lbl.gov/2025/reviews/rpp2024-rev-passage-particles-matter.pdf

The data values themselves are retrieved from https://pdg.lbl.gov/2025/AtomicNuclearProperties/
"""

from dataclasses import dataclass
from typing import Protocol

import hepunits as u
import jax.numpy as jnp

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
class StragglingParams:
    """Various parameters relevant to energy straggling"""

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
    is_atomic: bool
    """True if this is an atomic element (rather than a compound)"""
    density_correction: DensityCorrection

    def straggling_params(
        self, particle: IncidentParticle, thickness: SFloat
    ) -> StragglingParams:
        """Compute straggling parameters for a given particle and thickness"""

        beta, gamma = particle.beta(), particle.gamma()
        mass_ratio = ELECTRON_MASS * u.c_light_sq / particle.mass

        # maximum energy transfer in a single collision with an electron
        Wmax = (
            2
            * ELECTRON_MASS
            * u.c_light_sq
            * (beta * gamma) ** 2
            / (1 + 2 * gamma * mass_ratio + mass_ratio**2)
        )

        prefactor = (
            DEDX_CONSTANT * (self.Z / self.mass) * particle.charge**2 * self.density
        )

        xi = 0.5 * prefactor * thickness / beta**2
        kappa = xi / Wmax
        mean_energy_loss = xi * (
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
        # (0.2 is with density correction, 0.37 is without, per Bichsel:1998if)
        mode_energy_loss = mean_energy_loss + xi * (
            beta**2 + jnp.log(kappa) - 0.20005183774398613
        )
        return StragglingParams(
            xi=xi,
            kappa=kappa,
            mean_energy_loss=mean_energy_loss,
            mode_energy_loss=mode_energy_loss,
        )


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
        is_atomic=True,
        density_correction=DensityCorrection(
            C=4.2395, x0=0.1708, x1=3.0127, a=0.0802, k=3.6345, delta0=0.0
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
        is_atomic=False,
        density_correction=DensityCorrection(
            C=2.3580, x0=-0.0988, x1=1.4515, a=0.9057, k=2.5849, delta0=0.0
        ),
    ),
}
