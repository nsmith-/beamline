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
    """Compute elliptic integrals of the first, second, and third kind (K, E, Pi)

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
    """Shell radius [mm]"""
    jphi: float
    """Surface current density [e/ns]"""
    L: float
    """Length of the solenoid [mm]"""

    def Bz_onaxis(self, z, d=0):
        """Magnetic field on-axis (rho=0)

        d is the derivative order
        """
        halfL = self.L / 2
        left_term = _hypot_ratio(z + halfL, self.R, d=d)
        right_term = _hypot_ratio(z - halfL, self.R, d=d)
        Bz = MU0 * self.jphi / 2 * (left_term - right_term)
        return Bz

    def _B_rhoexpansion(self, rho, z, order=1):
        """Return the rho and z component of the magnetic field

        Uses a series expansion in rho around the on-axis value
        McDonald model, expanding in powers of rho around the on-axis field

        For the default order 1, this is slower than _B_wikipedia but probably
        this is due to poor optimization of the Bz derivative computation
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

    def _B_Caciagli(self, rho, z, rho_min=1e-7):
        """Return the rho and z component of the magnetic field

        Using the exact solution of 10.1016/j.nima.2022.166706
        TODO: check if this is actually less expensive than the Conway solution of 10.1109/20.947050

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

    def _B_wikipedia(self, rho, z, rho_min=1e-9):
        """Return the rho and z component of the magnetic field

        This is the same as _B_Caciagli but simplified by a very helpful Wikipedia editor
        It runs about 4x faster than the original, unfortunately it **DOES NOT AGREE**

        This formula is ill-defined for rho=0, so we clamp rho to a small value
        See _optimize_rho0limit in this source for details
        """
        rho = np.maximum(rho, rho_min)
        zetap, zetam = z + self.L / 2, z - self.L / 2
        _4Rrho = 4 * self.R * rho
        Rprho = self.R + rho
        n = _4Rrho / Rprho**2
        zrhypotp, zrhypotm = np.hypot(zetap, Rprho), np.hypot(zetam, Rprho)
        mp, mm = _4Rrho / zrhypotp**2, _4Rrho / zrhypotm**2
        Kp, Ep, Pip = _ellipkepi(n, mp)
        Km, Em, Pim = _ellipkepi(n, mm)
        prefactor = 0.5 * MU0 * self.jphi / np.pi
        Brho = (
            prefactor
            / (2 * rho)
            * (
                zrhypotp * (mp * Kp + 2 * (Ep - Kp))
                - zrhypotm * (mm * Km + 2 * (Em - Km))
            )
        )
        Bz = prefactor * (
            zetap / zrhypotp * (Kp + (self.R - rho) / Rprho * Pip)
            - zetam / zrhypotm * (Km + (self.R - rho) / Rprho * Pim)
        )
        return Brho, Bz

    def B(self, rho, z, rho_min=1e-7):
        """Return the rho and z component of the magnetic field

        Some method have issues at rho=0 so we clamp to a minimum rho
        """
        return self._B_Caciagli(rho, z, rho_min=rho_min)

    def plot_field(self, ax):
        rho, z = np.meshgrid(
            np.linspace(0, self.R * 2, 100),
            np.linspace(-self.L, self.L, 201),
            indexing="ij",
        )

        Brho, Bz = self.B(rho, z)

        contour = ax.contourf(z, rho, np.hypot(Brho, Bz), levels=100, cmap="RdBu_r")
        downsample = 5
        arrows = ax.quiver(
            z[::downsample, ::downsample],
            rho[::downsample, ::downsample],
            Bz[::downsample, ::downsample],
            Brho[::downsample, ::downsample],
            angles="xy",
        )
        outline = ax.plot(
            [-self.L / 2, -self.L / 2, self.L / 2, self.L / 2],
            [0, self.R, self.R, 0],
            "k--",
            lw=2,
        )
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("rho (mm)")
        return (contour, arrows, outline)


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
        Brho2, Bz2 = solenoid.B(rho, zpts, rho_min=0)
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

    Bz_axis = solenoid.Bz_onaxis(zpts)

    Brho, Bz = solenoid._B_Caciagli(0.0, zpts)
    assert np.allclose(Bz_axis, Bz), "Bz axial does not match the exact solution"
    assert np.allclose(Brho, 0.0), "Brho should be zero on-axis"

    Brho, Bz = solenoid._B_wikipedia(0.0, zpts)
    assert np.allclose(Bz_axis, Bz), "Bz axial does not match the Wikipedia solution"
    # assert np.allclose(Brho, 0.0), "Brho should be zero on-axis"

    Brho, Bz = solenoid._B_rhoexpansion(0.0, zpts)
    assert np.allclose(Bz_axis, Bz), "Bz axial does not match the expansion solution"
    assert np.allclose(Brho, 0.0), "Brho should be zero on-axis"

    rng = np.random.Generator(np.random.PCG64(42))
    zpts = rng.uniform(-100, 100, 10)
    rhopts = rng.uniform(0, 10, 10)

    with np.errstate(all="raise"):
        Brho, Bz = solenoid._B_wikipedia(rhopts, zpts)
        Brho2, Bz2 = solenoid._B_Caciagli(rhopts, zpts)
        Brho3, Bz3 = solenoid._B_rhoexpansion(rhopts, zpts)

    assert np.allclose(Brho, Brho2), "Brho does not match"
    assert np.allclose(Bz, Bz2), "Bz does not match"
    assert np.allclose(Brho, Brho3), "Brho does not match"
    assert np.allclose(Bz, Bz3), "Bz does not match"
