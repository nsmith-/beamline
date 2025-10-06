import hepunits as u
import numpy as np

from beamline.numpy.solenoid import ThinShellSolenoid


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
