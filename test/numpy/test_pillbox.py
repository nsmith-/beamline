from itertools import product

import hepunits as u
import numpy as np
import pytest
import vector

from beamline.numpy.pillbox import PillboxCavity
from beamline.units import check_dimensionality, ureg


def test_check_pillbox_units():
    # E0 definition is [MeV/e/mm]
    assert check_dimensionality(MeV=1, mm=-1, e=-1, expected=ureg.megavolt / ureg.meter)
    # B field is E/c
    assert (1 * ureg.megavolt / ureg.meter / ureg.c).to(ureg.tesla)
    # Test example equation
    Bphi = 1.0 * ureg.megavolt / ureg.meter * ureg.megahertz * ureg.meter / ureg.c**2
    assert Bphi.to(ureg.tesla)


def test_construct():
    tm010 = PillboxCavity(
        length=2 * u.m,
        radius=0.5 * u.m,
        E0=5 * u.megavolt / u.m,
        mode="TM",
        m=0,
        n=1,
        p=0,
        phase=0.0,
    )
    # TODO: do some tests with simplified formulas for TM010
    assert tm010.frequency / u.megahertz == pytest.approx(229.48505567042017)


def test_pillbox_surface_fields():
    """Test that the field on the surface is as expected, namely transverse E and normal B are zero"""

    for m, n, p, mode in product(range(4), range(1, 4), range(3), ("TE", "TM")):
        if n == 0:
            continue
        if mode == "TE" and p == 0:
            continue
        cavity = PillboxCavity(
            length=1 * u.m,
            radius=1 * u.m,
            E0=-5 * u.megavolt / u.m,
            mode=mode,
            m=m,
            n=n,
            p=p,
            phase=0.0,
        )

        ntests = 100

        # disk part
        for _ in range(ntests):
            phi = np.random.uniform(-np.pi, np.pi)
            rho = np.random.uniform(0, cavity.radius)
            z = np.random.choice([0, cavity.length])
            t = np.random.uniform(0, cavity.wavelength)
            pos = vector.obj(rho=rho, phi=phi, z=z, t=t)
            field = cavity.field_strength(pos)
            assert field.E.rho == pytest.approx(0.0)
            assert field.B.z == pytest.approx(0.0)

        # cylinder part
        for _ in range(ntests):
            phi = np.random.uniform(-np.pi, np.pi)
            rho = cavity.radius
            phihat = vector.VectorObject2D(rho=1, phi=phi + np.pi / 2)
            rhohat = vector.VectorObject2D(rho=1, phi=phi)
            z = np.random.uniform(0, cavity.length)
            t = np.random.uniform(0, cavity.wavelength)
            pos = vector.obj(rho=rho, phi=phi, z=z, t=t)
            field = cavity.field_strength(pos)
            assert field.E.z == pytest.approx(0.0)
            Ephihat = field.E.to_2D().dot(phihat)
            assert Ephihat == pytest.approx(0.0)
            Brhohat = field.B.to_2D().dot(rhohat)
            assert Brhohat == pytest.approx(0.0)
