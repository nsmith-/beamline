"""Four-potentials: the Lorentz force from ``contract_potential`` matches ``__call__``"""

import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.emfield import (
    EMTensorField,
    SimpleEMField,
    StackedField,
    TransformEMField,
)
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.magnet.solenoid import ThickSolenoid, ThinShellSolenoid

THICK = ThickSolenoid(
    Rin=250.0 * u.mm, Rout=419.3 * u.mm, jphi=500.0 * u.A / u.mm**2, L=140.0 * u.mm
)
LOOP = WireLoop(R=200.0 * u.mm, I=1.0e5 * u.A)
FIELDS = {
    "simple": SimpleEMField(
        E0=Cartesian3.make(x=3.0, y=-1.0, z=2.0) * (u.megavolt / u.m),
        B0=Cartesian3.make(x=0.2, y=-0.4, z=1.5) * u.tesla,
    ),
    "loop": LOOP,
    "thin": ThinShellSolenoid(R=300.0 * u.mm, jphi=50.0 * u.A / u.mm, L=140.0 * u.mm),
    "thick": THICK,
    "transformed": TransformEMField(
        transform=Transform.make_axis_angle(
            Cartesian3.make(x=1.0, y=1.0), 0.3, Cartesian4.make(x=20.0, z=-50.0)
        ),
        field=THICK,
    ),
    "sum": LOOP + THICK,
    "stacked": StackedField.stack(
        [
            WireLoop(R=200.0 * u.mm, I=1.0e5 * u.A),
            WireLoop(R=150.0 * u.mm, I=-3.0e4 * u.A),
        ]
    ),
}


def _points() -> Cartesian4:
    """Generic points, plus points on and very near the axis"""
    rng = np.random.default_rng(42)
    generic = rng.uniform(-150.0, 150.0, size=(16, 4))
    axis = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 60.0, 1.0],
            [1e-7, 0.0, -30.0, 0.0],
            [0.0, 2e-3, 10.0, 0.0],
        ]
    )
    return Cartesian4(coords=jnp.asarray(np.concatenate([generic, axis])))


def _momenta(n: int) -> Cartesian4:
    rng = np.random.default_rng(7)
    p3 = rng.normal(scale=30.0, size=(n, 3)) + np.array([0.0, 0.0, 200.0])
    energy = np.sqrt(np.sum(p3**2, axis=-1) + 105.66**2)
    return Cartesian4(coords=jnp.asarray(np.concatenate([p3, energy[:, None]], -1)))


@pytest.mark.parametrize("name", list(FIELDS))
def test_contract_potential(name: str):
    field: EMTensorField = FIELDS[name]
    points = _points()
    vec = Tangent(p=points, t=_momenta(points.coords.shape[0]))

    expected = jax.vmap(field)(vec).t.coords
    actual = jax.vmap(field.contract_potential)(vec).t.coords
    scale = jnp.max(jnp.abs(expected))
    assert actual == pytest.approx(np.asarray(expected), rel=1e-8, abs=1e-9 * scale)


@pytest.mark.parametrize("name", ["loop", "thin", "thick"])
def test_contract_potential_jacobian_on_axis(name: str):
    """The force's Jacobian (what a transfer map integrates) is finite on axis"""
    field: EMTensorField = FIELDS[name]
    p = _momenta(1).coords[0]

    def force(x):
        vec = Tangent(p=Cartesian4(coords=x), t=Cartesian4(coords=p))
        return field.contract_potential(vec).t.coords

    def force_ref(x):
        vec = Tangent(p=Cartesian4(coords=x), t=Cartesian4(coords=p))
        return field(vec).t.coords

    x = jnp.array([0.0, 0.0, 30.0, 0.0])
    jac = jax.jacfwd(force)(x)
    assert jnp.all(jnp.isfinite(jac))
    jac_ref = jax.jacfwd(force_ref)(x)
    scale = jnp.max(jnp.abs(jac_ref))
    assert jac == pytest.approx(np.asarray(jac_ref), abs=1e-7 * scale)
