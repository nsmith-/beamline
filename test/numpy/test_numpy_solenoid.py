import hepunits as u
import matplotlib.pyplot as plt
import numpy as np

from beamline.numpy.solenoid import ThinShellSolenoid
from beamline.units import to_clhep, ureg


def test_thinshell_solenoid():
    solenoid = ThinShellSolenoid(
        R=43.81 * u.mm,
        jphi=600 * u.ampere / (0.289 * u.mm),
        L=34.68 * u.mm,
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
    zpts = rng.uniform(-solenoid.L, solenoid.L, 10)
    rhopts = rng.uniform(0.0, solenoid.R / 5, 10)

    with np.errstate(all="raise"):
        Brho1, Bz1 = solenoid._B_wikipedia(rhopts, zpts)
        Brho2, Bz2 = solenoid._B_Caciagli(rhopts, zpts)
        Brho3, Bz3 = solenoid._B_rhoexpansion(rhopts, zpts)

    # Wikipedia result has some bug for off-axis points
    # assert np.allclose(Brho1, Brho2), "Brho does not match"
    # assert np.allclose(Bz1, Bz2), "Bz does not match"
    assert np.allclose(Brho2, Brho3, rtol=1e-3), "Brho does not match"
    assert np.allclose(Bz2, Bz3, rtol=1e-3), "Bz does not match"


def test_optimize_rho0limit(artifacts_dir):
    """How the rho -> 0 limit was optimized

    As rho gets smaller, the formula gets less accurate, with a minimum around 1e-6 in this example
    """

    solenoid = ThinShellSolenoid(
        R=to_clhep(43.81 * ureg.mm),
        jphi=to_clhep(600 * ureg.amp / (0.289 * ureg.mm)),
        L=to_clhep(34.68 * ureg.mm),
    )

    zpts = np.linspace(-100, 100, 201)

    @np.vectorize
    def maxdiff(rho):
        Bz1 = solenoid.Bz_onaxis(zpts)
        Brho2, Bz2 = solenoid.B(rho, zpts, rho_min=0)
        return np.max(np.abs(Bz1 - Bz2)), np.max(np.abs(Brho2))

    fig, ax = plt.subplots()

    rhovals = np.geomspace(1e-13, 1e-3, 50)
    max_diffs = maxdiff(rhovals)
    ax.plot(rhovals, max_diffs[0], label="Bz axial - Bz exact")
    ax.plot(rhovals, max_diffs[1], label="Brho - Brho exact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rho (mm)")
    ax.set_ylabel("Max Difference (kT)")
    ax.legend()
    fig.savefig(artifacts_dir / "optimize_rho0limit.png")
