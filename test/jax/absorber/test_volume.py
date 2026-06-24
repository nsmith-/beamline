"""Tests for the material volume geometry (``AbsorberCylinder``)."""

import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder
from beamline.jax.coordinates import Cartesian3, Tangent

ABSORBER_RADIUS = 100.0 * u.mm
ABSORBER_LENGTH = 100.0 * u.mm


def make_absorber(char_length: float = 10.0 * u.mm) -> AbsorberCylinder:
    return AbsorberCylinder(
        material=MATERIALS["lithium_hydride_LiH"],
        radius=ABSORBER_RADIUS,
        length=ABSORBER_LENGTH,
        char_length=char_length,
    )


def test_absorber_geometry():
    """contains/signed_distance match the expected cylindrical bounds."""
    absorber = make_absorber()

    assert bool(absorber.contains(Cartesian3.make()))  # origin, inside
    assert not bool(absorber.contains(Cartesian3.make(z=100.0 * u.mm)))  # past end
    assert not bool(absorber.contains(Cartesian3.make(x=150.0 * u.mm)))  # past radius

    # On-axis ray approaching the front face (at z = -length/2 = -50 mm) from
    # z = -200 mm at unit speed: the nearest surface is 150 mm ahead.
    entering = Tangent(p=Cartesian3.make(z=-200.0 * u.mm), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_distance(entering)) == pytest.approx(
        150.0 * u.mm, rel=1e-6
    )

    # A ray offset beyond the radius, parallel to the axis, never enters.
    missing = Tangent(
        p=Cartesian3.make(x=150.0 * u.mm, z=-200.0 * u.mm), t=Cartesian3.make(z=1.0)
    )
    assert jnp.isinf(absorber.signed_distance(missing))
