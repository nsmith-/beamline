"""Passage of particles through material

These are energy loss and straggling effect computations,
as described e.g. in https://pdg.lbl.gov/2025/reviews/rpp2024-rev-passage-particles-matter.pdf

The data values themselves are retrieved from https://pdg.lbl.gov/2025/AtomicNuclearProperties/
"""

from dataclasses import dataclass
from typing import ClassVar

import hepunits as u
import numpy as np

from beamline.units import to_clhep, ureg


@dataclass
class Material:
    """A material that particles may pass through"""

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

    def density_correction(self, bg: float):
        """Density correction delta(beta*gamma)

        Sternheimer's parameterization
        """
        return 0.0  # TODO


@dataclass
class IncidentParticle:
    """Used just to simplify material interaction code"""

    mass: float
    """Incident particle mass-energy [MeV]"""
    z: int
    """Incident particle charge [e]"""

    def bg(self, pc: float):
        """Relativistic beta = v/c and gamma = E / mc^2

        Args:
            pc: momentum times c [MeV]
        """
        E = np.hypot(self.mass, pc)
        return pc / E, E / self.mass


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
    ),
}


PARTICLES: dict[str, IncidentParticle] = {
    "muon": IncidentParticle(
        mass=to_clhep(ureg.muon_mass) * u.c_light_sq,
        z=1,
    ),
    "pion": IncidentParticle(
        mass=to_clhep(ureg.pion_mass) * u.c_light_sq,
        z=1,
    ),
    "electron": IncidentParticle(
        mass=to_clhep(ureg.electron_mass) * u.c_light_sq,
        z=1,
    ),
}

ele = PARTICLES["electron"]
"shorthand for electron since we use it a lot"


@dataclass
class StragglingParams:
    """Various parameters relevant to energy straggling"""

    xi: float
    """Landau's xi (the scaling of the dimensionless Landau parameter)"""
    kappa: float
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
    mean_energy_loss: float
    """Mean energy loss (Bethe-Bloch formula)"""
    landau_mean: float
    """Landau parameter shift"""
    delta_p: float
    """Most probable energy loss"""


@dataclass
class MaterialInteraction:
    material: Material
    particle: IncidentParticle
    K: ClassVar[float] = 0.307075 * u.MeV * u.cm2 / u.mol
    r"""<dE/dx> coefficient from PDG

    Equal to $4 \pi N_A r_e^2 m_e c^2$
    where $r_e$ is the classical electron radius,
    $m_e$ is the electron mass, and $N_A$ is Avogadro's number.
    """
    Bichsel_k: ClassVar[float] = 2.5496e-19 * u.eV * u.cm2
    r"""Bichsel's k constant

    Equal to K / 2 / Avogadro's number
    """

    @property
    def prefactor(self):
        """Common prefactor for many formulas

        Units: [MeV / mm]
        """
        return (
            self.K
            * (self.material.Z / self.material.mass)
            * self.particle.z**2
            * self.material.density
        )

    def Wmax(self, pc: float):
        """Maximum energy transfer in a single collision

        Args:
            pc: particle momentum times c [MeV]
        """
        beta, gamma = self.particle.bg(pc)
        mratio = ele.mass / self.particle.mass
        return 2 * ele.mass * (beta * gamma) ** 2 / (1 + 2 * gamma * mratio + mratio**2)

    def w(self, pc: float, E: float):
        r"""Probability of particle with momentum pc scattering off free electron losing energy E per unit length

        Units: [mm^-1 MeV^-1]

        See Eqn. 3.5, 3.6 of Bichsel:1988if

        Other references: Hancock:1983fp, Landau, and Vavilov
        """
        # Landau:
        # return prefactor / (beta(p)*E)**2
        # Vavilov:
        beta, _gamma = self.particle.bg(pc)
        Wmax = self.Wmax(pc)
        # Landau term
        wval = self.prefactor / (beta * E) ** 2
        # Vavilov correction (finite maximum energy transfer)
        wval = wval * (1 - beta**2 * E / Wmax)
        # TODO: handle density correction
        # delta = self.material.density_correction(beta * gamma)
        return np.where(
            self.material.mean_excitation > E, 0.0, np.where(Wmax > E, wval, 0.0)
        )

    def w_norm(self, pc):
        """Total probability of particle scattering with E > mean excitation"""
        beta, _ = self.particle.bg(pc)
        Wmax = self.Wmax(pc)
        landau_norm = (
            self.prefactor / beta**2 * (1 / self.material.mean_excitation - 1 / Wmax)
        )
        vavilov_corr = (
            -self.prefactor / Wmax * (np.log(Wmax / self.material.mean_excitation))
        )
        return landau_norm + vavilov_corr

    def dEdx_mean_BetheBloch(self, pc: float):
        """Bethe Bloch mean energy loss dE/dx approximation"""
        beta, gamma = self.particle.bg(pc)
        bg = beta * gamma
        Wmax = self.Wmax(pc)
        return (
            self.prefactor
            / beta**2
            * (
                0.5
                * np.log(2 * ele.mass * bg**2 * Wmax / self.material.mean_excitation**2)
                - beta**2
                - 0.5 * self.material.density_correction(bg)
            )
        )

    def straggling_params(self, pc: float, thickness: float) -> StragglingParams:
        """Landau/Vavilov/Shulek parameters"""
        beta, gamma = self.particle.bg(pc)
        xi = 0.5 * self.prefactor * thickness / beta**2
        kappa = xi / self.Wmax(pc)
        mean_Delta = self.dEdx_mean_BetheBloch(pc) * thickness
        mean_lambda = -(1 - np.euler_gamma) - beta**2 - np.log(kappa)
        landau_loc = mean_Delta - xi * mean_lambda
        delta_p = xi * (
            np.log(2 * ele.mass * (beta * gamma) ** 2 / self.material.mean_excitation)
            + np.log(xi / self.material.mean_excitation)
            # + 0.2  # TODO: add when density correction is in
            - beta**2
            - self.material.density_correction(beta * gamma)
        )
        return StragglingParams(
            xi=xi,
            kappa=kappa,
            mean_energy_loss=mean_Delta,
            landau_mean=landau_loc,
            delta_p=delta_p,
        )
