import hepunits as u
import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian4
from beamline.jax.emfield import EMTensorField
from beamline.jax.magnet.axial_expansion import AXIS_SERIES_RADIUS, B_rhoexpansion
from beamline.jax.magnet.loop import WireLoop
from beamline.jax.magnet.solenoid import ThickSolenoid, ThinShellSolenoid

FIELDS = {
    "loop": WireLoop(R=100.0 * u.mm, I=50.0 * u.A),
    "thin": ThinShellSolenoid(R=100.0 * u.mm, jphi=1.0, L=200.0 * u.mm),
    "thick": ThickSolenoid(
        Rin=250.0 * u.mm, Rout=419.3 * u.mm, jphi=500.0 * u.A / u.mm**2, L=140.0 * u.mm
    ),
}


def _Bcart(field: EMTensorField):
    def f(xyz):
        _, B = field.field_strength(Cartesian4.make(x=xyz[0], y=xyz[1], z=xyz[2]))
        return B.t.coords

    return f


@pytest.mark.parametrize("name", FIELDS)
def test_field_strength_on_axis(name: str):
    """Exactly on axis the field and its Jacobian are finite, with div B = 0"""
    Bcart = _Bcart(FIELDS[name])
    for z in (0.0, 30.0 * u.mm, -150.0 * u.mm):
        p = jnp.array([0.0, 0.0, z])
        B = Bcart(p)
        assert jnp.all(jnp.isfinite(B))
        assert B[0] == 0.0
        assert B[1] == 0.0
        J = jax.jacfwd(Bcart)(p)
        assert jnp.all(jnp.isfinite(J))
        scale = jnp.abs(J).max()
        if z == 0.0:
            # mid-plane symmetry: Brho = 0 and dBz/dz = 0, so the Jacobian vanishes
            assert scale == 0.0
            continue
        assert float(jnp.trace(J)) == pytest.approx(0.0, abs=1e-12 * float(scale))
        # transverse gradient is isotropic, and Bz has no transverse gradient on axis
        assert float(J[0, 0]) == pytest.approx(float(J[1, 1]), rel=1e-12)
        assert float(J[2, 0]) == pytest.approx(0.0, abs=1e-12 * float(scale))


def _full_B(field: EMTensorField):
    if isinstance(field, WireLoop):
        return field.B
    if isinstance(field, ThinShellSolenoid):
        return field.B_elliptic
    return field.B_shells


def _size(field: EMTensorField) -> float:
    return float(field.Rin if isinstance(field, ThickSolenoid) else field.R)


@pytest.mark.parametrize("name", FIELDS)
def test_field_strength_matches_cylindric(name: str):
    """Away from the axis, field_strength is exactly the full cylindrical field"""
    field = FIELDS[name]
    B = _full_B(field)
    for x, y, z in [(30.0, -40.0, 10.0), (80.0, 20.0, -60.0), (0.3, 0.0, 5.0)]:
        rho = jnp.hypot(x, y)
        Brho, Bz = B(rho, z)
        expected = jnp.array([Brho * x / rho, Brho * y / rho, Bz])
        got = _Bcart(field)(jnp.array([x, y, z]))
        assert jnp.allclose(got, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("name", FIELDS)
def test_switch_continuity(name: str):
    """The rho expansion and the full field agree where field_strength switches"""
    field = FIELDS[name]
    r = AXIS_SERIES_RADIUS * _size(field)
    zs = jnp.linspace(-300.0 * u.mm, 300.0 * u.mm, 61)

    def Bcart(rho, z):
        _, B = field.field_strength(Cartesian4.make(x=rho, z=z))
        return B.t.coords

    inside = jax.vmap(Bcart, in_axes=(None, 0))(r * (1 - 1e-12), zs)
    outside = jax.vmap(Bcart, in_axes=(None, 0))(r * (1 + 1e-12), zs)
    # Bx = Brho is ~ r / L smaller than Bz here, so compare each to its own scale
    for i in (0, 2):
        scale = jnp.max(jnp.abs(outside[:, i]))
        assert jnp.max(jnp.abs(inside[:, i] - outside[:, i])) < 1e-11 * scale


def test_rhoexpansion_hypot_corner():
    """The on-axis derivatives are right where |z +- L/2| == R

    jnp.hypot has wrong second derivatives at |x| == |y|; here z = 0 with L = 2R.
    """
    field = FIELDS["thin"]
    rho = 0.01 * field.R
    _, Bz_exp = B_rhoexpansion(field.Bz_onaxis, rho, 0.0, order=3)
    _, Bz = field.B_elliptic(rho, 0.0)
    assert float(Bz_exp) == pytest.approx(float(Bz), rel=1e-12)
