import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.emfield import EMTensorField, StackedField, TransformEMField
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.rfcavity.pillbox import PillboxCavity

POINT = Cartesian4.make(x=30.0 * u.mm, y=-10.0 * u.mm, z=350.0 * u.mm, ct=1.3 * u.m)
VEC = Tangent(
    p=POINT,
    t=Cartesian4.make(x=1.0 * u.MeV, y=2.0 * u.MeV, z=200.0 * u.MeV, ct=210.0 * u.MeV),
)


def _cavity() -> PillboxCavity:
    return PillboxCavity(
        length=200.0 * u.mm,
        frequency=0.805 * u.GHz,
        E0=15.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=1,
        phase=0.3,
    )


def _transform() -> Transform:
    return Transform.make_axis_angle(
        Cartesian3.make(x=1.0, y=2.0, z=0.5),
        0.7,
        Cartesian4.make(x=20.0 * u.mm, y=-15.0 * u.mm, z=300.0 * u.mm, ct=0.4 * u.m),
    )


@pytest.mark.parametrize(
    "field",
    [_cavity(), WireLoop(R=250.0 * u.mm, I=50.0 * u.A)],
    ids=["cavity", "loop"],
)
def test_field_strength_matches_call(field: EMTensorField):
    """field_strength contracted by the base class must equal the transformed __call__"""
    tf = TransformEMField(transform=_transform(), field=field)
    via_fields = EMTensorField.__call__(tf, VEC)
    via_call = tf(VEC)
    assert jnp.linalg.norm(via_call.t.coords) > 0.0
    assert jnp.allclose(via_fields.t.coords, via_call.t.coords, rtol=1e-10)


def test_field_strength_in_stack():
    tfs = [TransformEMField(transform=_transform(), field=_cavity())] * 2
    stacked = StackedField.stack(tfs)
    E, B = stacked.field_strength(POINT)
    E1, B1 = tfs[0].field_strength(POINT)
    assert jnp.allclose(E.t.coords, 2 * E1.t.coords, rtol=1e-10)
    assert jnp.allclose(B.t.coords, 2 * B1.t.coords, rtol=1e-10)
    assert jnp.linalg.norm(E1.t.coords) + jnp.linalg.norm(B1.t.coords) > 0.0
