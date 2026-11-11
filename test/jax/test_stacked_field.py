import equinox as eqx
import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.emfield import StackedField, SumField, TransformEMField
from beamline.jax.magnet.solenoid import ThinShellSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity


def _placed(z0: float) -> TransformEMField:
    return TransformEMField(
        transform=Transform.make_translation(z=z0),
        field=ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=200.0 * u.mm),
    )


def _placed_cavity(z0: float) -> TransformEMField:
    return TransformEMField(
        transform=Transform.make_translation(z=z0),
        field=PillboxCavity(
            length=200.0 * u.mm,
            frequency=0.805 * u.GHz,
            E0=15.0 * u.MV / u.m,
            mode="TM",
            m=0,
            n=1,
            p=1,
            phase=0.0,
        ),
    )


Z0S = [0.0, 300.0 * u.mm, 600.0 * u.mm]
POINT = Cartesian4.make(x=30.0 * u.mm, y=-10.0 * u.mm, z=350.0 * u.mm)
VEC = Tangent(
    p=POINT,
    t=Cartesian4.make(x=1.0 * u.MeV, y=2.0 * u.MeV, z=200.0 * u.MeV, ct=210.0 * u.MeV),
)


def test_stack_matches_vmap_construction():
    zs = jnp.array(Z0S)
    vmapped = StackedField(fields=eqx.filter_vmap(_placed)(zs))
    stacked = StackedField.stack([_placed(z) for z in Z0S])
    assert len(vmapped) == len(stacked) == 3
    out1, out2 = vmapped(VEC), stacked(VEC)
    assert jnp.linalg.norm(out1.t.coords) > 0.0
    assert jnp.allclose(out1.t.coords, out2.t.coords, rtol=1e-12)


def test_matches_sumfield_call():
    stacked = StackedField.stack([_placed(z) for z in Z0S])
    summed = SumField([_placed(z) for z in Z0S])
    assert jnp.allclose(stacked(VEC).t.coords, summed(VEC).t.coords, rtol=1e-12)


def test_field_strength_matches_sumfield():
    # TransformEMField.field_strength is unavailable by design, so use bare members
    def shell(jphi: float) -> ThinShellSolenoid:
        return ThinShellSolenoid(R=100.0 * u.mm, jphi=jphi, L=200.0 * u.mm)

    jphis = [1.0, -0.5, 2.0]
    stacked = StackedField.stack([shell(j) for j in jphis])
    summed = SumField([shell(j) for j in jphis])
    E1, B1 = stacked.field_strength(POINT)
    E2, B2 = summed.field_strength(POINT)
    assert jnp.linalg.norm(B1.t.coords) > 0.0
    assert jnp.allclose(E1.t.coords, E2.t.coords, rtol=1e-12)
    assert jnp.allclose(B1.t.coords, B2.t.coords, rtol=1e-12)


def test_unstack_roundtrip():
    stacked = StackedField.stack([_placed(z) for z in Z0S])
    parts = stacked.unstack()
    assert len(parts) == 3
    for part, z in zip(parts, Z0S, strict=True):
        assert jnp.allclose(part(VEC).t.coords, _placed(z)(VEC).t.coords, rtol=1e-12)


def test_geometry():
    cavs = [_placed_cavity(z) for z in Z0S]
    stacked = StackedField.stack(cavs)
    # Inside the middle cavity, and far outside all of them
    inside = Cartesian3.make(z=300.0 * u.mm)
    outside = Cartesian3.make(x=500.0 * u.mm)
    assert bool(stacked.contains(inside))
    assert not bool(stacked.contains(outside))

    summed = SumField(cavs)
    for z in (-500.0 * u.mm, 150.0 * u.mm, 300.0 * u.mm):
        ray = Tangent(p=Cartesian3.make(z=z), t=Cartesian3.make(z=1.0))
        assert float(stacked.signed_time_to_boundary(ray)) == pytest.approx(
            float(summed.signed_time_to_boundary(ray))
        )
