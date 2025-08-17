"""Solenoid models

Most are coming from the references
https://doi.org/10.1016/j.nima.2022.166706
https://doi.org/10.1109/20.947050
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import elliprd, elliprf, elliprj, factorial

from beamline.units import to_clhep, u

MU0 = to_clhep(1 * u.vacuum_permeability)


def _hypot_ratio(x, y, d=0):
    """Calculate x/sqrt(x**2+y**2) or its derivatives"""
    hypot = np.hypot(x, y)
    if d == 0:
        return x / hypot
    elif d == 1:
        return y**2 / hypot**3
    elif d == 2:
        return -3 * x / hypot**3 + 3 * x**3 / hypot**5
    elif d == 3:
        return -3 / hypot**3 + 18 * x**2 / hypot**5 - 15 * x**4 / hypot**7
    elif d == 4:
        return 45 * x / hypot**5 - 150 * x**3 / hypot**7 + 105 * x**5 / hypot**9
    raise NotImplementedError("Higher derivatives are not implemented")


def _ellipkepi(n, k):
    """Compute elliptic integrals of the first, second, and third kind

    Doing it once using Carlson forms may save a little bit of time
    https://en.wikipedia.org/wiki/Carlson_symmetric_form#Complete_elliptic_integrals
    """
    Rf = elliprf(0, 1 - k**2, 1)
    Rd = elliprd(0, 1 - k**2, 1)
    Rj = elliprj(0, 1 - k**2, 1, 1 - n)
    K = Rf
    E = Rf - k**2 / 3 * Rd
    Pi = Rf + n / 3 * Rj
    return K, E, Pi


def _auxp12(k2, gamma):
    """Auxiliary functions for implementing 10.1109/20.947050

    From Eqn. 4, 6"""
    sqrt1mk2 = np.sqrt(1 - k2)
    _1mgamma2 = 1 - gamma**2
    K, E, Pi = _ellipkepi(_1mgamma2, sqrt1mk2)
    P1 = K - 2 * (K - E) / (1 - k2)
    P2 = -gamma / _1mgamma2 * (Pi - K) - 1 / _1mgamma2 * (gamma**2 * Pi - K)
    return P1, P2


@dataclass
class ThinShellSolenoid:
    r"""Solenoid

    The thin-shell solenoid model on-axis ($\rho=0$) has the field:

    $$
    B_z(\rho=0,z) = \frac{\mu_0 j_\phi}{2} \left( \frac{z + L/2}{\sqrt{R^2 + (z+L/2)^2}} - \frac{z - L/2}{\sqrt{R^2 + (z-L/2)^2}}  \right)
    $$

    This is derived by integrating the on-axis wire loop model along the $z$ direction. We could also use this wire loop integral to model arbitrary $j_\phi = j(z)$.

    To derive the off-axis solution, we can use an expansion in $\rho$ (McDonald model, 10.1016/j.nima.2022.166706, Eqn. 10-12):

    $$
    B_z(\rho,z) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2} \frac{\partial^{2n} B_{z}(0,z)}{\partial z^{2n}} (\rho/2)^{2n}
    $$

    or we can use an exact solution in terms of elliptic integrals
    """

    R: float
    """Shell radius"""
    jphi: float
    """Surface current density"""
    L: float
    """Length of the solenoid"""

    def Bz_onaxis(self, z, d=0):
        """Magnetic field on-axis (rho=0)

        d is the derivative order
        """
        halfL = self.L / 2
        left_term = _hypot_ratio(z + halfL, self.R, d=d)
        right_term = _hypot_ratio(z - halfL, self.R, d=d)
        Bz = MU0 * self.jphi / 2 * (left_term - right_term)
        return Bz

    def B_expansion(self, rho, z, order=1):
        """Return the rho and z component of the magnetic field

        Uses a series expansion in rho around the on-axis value
        McDonald model, expanding in powers of rho around the on-axis field

        For the default order 1, this is roughly 10x faster than the exact solution
        But it catches up quickly in time, probably due to poor design of the Bz derivatives
        TODO: improve Bz derivatives computation
        """
        halfRho = rho / 2
        Bz = self.Bz_onaxis(z)
        Brho = -self.Bz_onaxis(z, d=1) * halfRho
        # TODO: when jax-ifying, this is a good place for a jet
        for n in range(1, order + 1):
            prefactor = (-1) ** n / factorial(n) ** 2
            Bz += prefactor * self.Bz_onaxis(z, d=2 * n) * (halfRho ** (2 * n))
            prefactor *= -1 / (n + 1)
            Brho += (
                prefactor * self.Bz_onaxis(z, d=2 * n + 1) * (halfRho ** (2 * n + 1))
            )

        return Brho, Bz

    def B(self, rho, z, rho_min=1e-7):
        """Return the rho and z component of the magnetic field

        Using the exact solution of 10.1109/20.947050
        (maybe less expensive than the Conway solution of 10.1016/j.nima.2022.166706 ?)

        This formula is ill-defined for rho=0, so we clamp rho to a small value
        See _optimize_rho0limit in this source for details
        """
        rho = np.maximum(rho, rho_min)
        xip, xim = z + self.L / 2, z - self.L / 2
        rhopR, rhomR = rho + self.R, rho - self.R
        alphap, alpham = 1 / np.hypot(xip, rhopR), 1 / np.hypot(xim, rhopR)
        betap, betam = xip * alphap, xim * alpham
        gamma = rhomR / rhopR
        k2p, k2m = (
            (xip**2 + rhomR**2) / (xip**2 + rhopR**2),
            (xim**2 + rhomR**2) / (xim**2 + rhopR**2),
        )
        P1p, P2p = _auxp12(k2p, gamma)
        P1m, P2m = _auxp12(k2m, gamma)
        prefactor = MU0 * self.jphi * self.R / np.pi
        Brho = prefactor * (alphap * P1p - alpham * P1m)
        Bz = prefactor / rhopR * (betap * P2p - betam * P2m)
        return Brho, Bz


def _optimize_rho0limit():
    """How the rho -> 0 limit was optimized

    As rho gets smaller, the formula gets less accurate, with a minimum around 1e-6 in this example
    """
    import matplotlib.pyplot as plt

    solenoid = ThinShellSolenoid(
        R=to_clhep(43.81 * u.mm),
        jphi=to_clhep(600 * u.amp / (0.289 * u.mm)),
        L=to_clhep(34.68 * u.mm),
    )

    zpts = np.linspace(-100, 100, 201)

    @np.vectorize
    def maxdiff(rho):
        Bz1 = solenoid.Bz_onaxis(zpts)
        Brho2, Bz2 = solenoid.B(rho, zpts)
        return np.max(np.abs(Bz1 - Bz2)), np.max(np.abs(Brho2))

    fig, ax = plt.subplots()

    rhovals = np.geomspace(1e-14, 1e-3, 50)
    max_diffs = maxdiff(rhovals)
    ax.plot(rhovals, max_diffs[0], label="Bz axial - Bz exact")
    ax.plot(rhovals, max_diffs[1], label="Brho - Brho exact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rho (mm)")
    ax.set_ylabel("Max Difference (kT)")
    ax.legend()
    return fig


if __name__ == "__main__":
    solenoid = ThinShellSolenoid(
        R=to_clhep(43.81 * u.mm),
        jphi=to_clhep(600 * u.amp / (0.289 * u.mm)),
        L=to_clhep(34.68 * u.mm),
    )

    zpts = np.linspace(-100, 100, 201)

    Bz1 = solenoid.Bz_onaxis(zpts)
    Brho2, Bz2 = solenoid.B(1e-9, zpts)
    assert np.allclose(Bz1, Bz2), "Bz axial does not match the exact solution"
    assert np.allclose(Brho2, 0.0), "Brho should be zero on-axis"
