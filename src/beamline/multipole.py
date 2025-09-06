"""Multipole models


TODO: look at http://dx.doi.org/10.1103/PhysRevSTAB.18.064001
"""

from dataclasses import dataclass

import numpy as np



@dataclass
class IdealMultipole:
    """Ideal multipole model

    This is a model of an ideal multipole magnet
    """

    n: int
    """Multipole order (1: dipole, 2: quadrupole, 3: sextupole, etc.)"""
    k: float
    """Multiple strength [kT]"""
    phi: float = 0.0
    """Rotation angle [rad]"""

    def B(self, x, y):
        C = self.k * np.exp(1j * self.phi)
        Bcomplex = C * (x - 1j * y) ** (self.n - 1)
        Bx, By = Bcomplex.real, Bcomplex.imag
        return Bx, By

    def plot_field(self, ax, rmax=2):
        x, y = np.meshgrid(
            np.linspace(-rmax, rmax, 100),
            np.linspace(-rmax, rmax, 101),
            indexing="ij",
        )

        Bx, By = self.B(x, y)

        contour = ax.contourf(x, y, np.hypot(Bx, By), levels=100, cmap="RdBu_r")
        downsample = 5
        arrows = ax.quiver(
            x[::downsample, ::downsample],
            y[::downsample, ::downsample],
            Bx[::downsample, ::downsample],
            By[::downsample, ::downsample],
            angles="xy",
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        return (contour, arrows)
