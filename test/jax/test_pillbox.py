from itertools import product

import hepunits as u
import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import (
    Cartesian3,
    Cartesian4,
    Cylindric3,
    Cylindric4,
    DivergenceField,
)
from beamline.jax.pillbox import PillboxCavity


def test_pillbox_init():
    with pytest.raises(ValueError, match="n must be >= 1"):
        PillboxCavity(
            length=1 * u.m,
            frequency=704 * u.MHz,
            E0=30 * u.megavolt / u.m,
            mode="TM",
            m=0,
            n=0,
            p=1,
            phase=0.0,
        )

    with pytest.raises(ValueError, match="p must be >= 1 for TE modes"):
        PillboxCavity(
            length=1 * u.m,
            frequency=704 * u.MHz,
            E0=30 * u.megavolt / u.m,
            mode="TE",
            m=0,
            n=1,
            p=0,
            phase=0.0,
        )

    with pytest.raises(
        ValueError, match="Frequency is too low for the given length and p"
    ):
        PillboxCavity(
            length=1 * u.m,
            frequency=100 * u.MHz,  # too low for p=3 and L=1m
            E0=30 * u.megavolt / u.m,
            mode="TM",
            m=0,
            n=1,
            p=3,
            phase=0.0,
        )


def surface_samples(R, L, lam, ntests):
    """Generate random samples on the surface of a pillbox cavity for testing

    Args:
        R: Cavity radius [mm]
        L: Cavity length [mm]
        lam: Cavity wavelength [mm]
        ntests: Number of random samples to generate
    """
    keys = jax.random.split(jax.random.PRNGKey(234), 5)
    shape = (ntests,)

    # disk part
    phi = jax.random.uniform(keys[0], minval=-jnp.pi, maxval=jnp.pi, shape=shape)
    rho = jax.random.uniform(keys[1], minval=0.0, maxval=R, shape=shape)
    z = jax.random.choice(keys[2], jnp.array([-L / 2, L / 2]), shape=shape)
    ct = jax.random.uniform(keys[3], minval=0.0, maxval=lam, shape=shape)
    pos = Cylindric4.make(rho=rho, phi=phi, z=z, ct=ct)

    # cylinder part
    z_cyl = jax.random.uniform(keys[4], minval=-L / 2, maxval=L / 2, shape=shape)
    pos_cyl = Cylindric4.make(rho=jnp.full_like(rho, R), phi=phi, z=z_cyl, ct=ct)

    return pos.to_cartesian(), pos_cyl.to_cartesian()


def interior_samples(R, L, ntests):
    keys = jax.random.split(jax.random.PRNGKey(345), 4)
    shape = (ntests,)
    phi = jax.random.uniform(keys[0], minval=-jnp.pi, maxval=jnp.pi, shape=shape)
    rho = jax.random.uniform(keys[1], minval=0.0, maxval=R, shape=shape)
    z = jax.random.uniform(keys[2], minval=-L / 2, maxval=L / 2, shape=shape)
    pos = Cylindric3.make(rho=rho, phi=phi, z=z)
    return pos.to_cartesian()


@pytest.mark.parametrize(
    ("m", "n", "p", "mode"),
    [
        tup
        for tup in product(range(3), range(1, 3), range(3), ("TE", "TM"))
        if not (tup[3] == "TE" and tup[2] == 0)
    ],
)
def test_jax_pillbox(m, n, p, mode):
    """Test some expected behaviours for the pillbox

    - the transverse E and normal B field are zero on the surface
    - the divergence of E and B are zero in the interior
    Bonus TODO: Maxwell's equations: del x E = -dB/dt and del x B = dE/dt in the interior
    """
    ntests = 100

    cavity = PillboxCavity(
        length=1 * u.m,
        frequency=704 * u.MHz,
        E0=30 * u.megavolt / u.m,
        mode=mode,
        m=m,
        n=n,
        p=p,
        phase=0.0,
    )

    pos_end, pos_cyl = surface_samples(
        cavity.radius, cavity.length, cavity.wavelength, ntests
    )
    zero = pytest.approx(jnp.zeros_like(pos_end.ct), abs=1e-10)

    E, B = jax.vmap(cavity.field_strength)(pos_end)
    assert E.t.rho == zero
    assert B.t.z == zero

    E, B = jax.vmap(cavity.field_strength)(pos_cyl)
    # here better to be in cylindric tangent basis
    E = E.to_cylindric()
    B = B.to_cylindric()
    assert E.t.z == zero
    assert E.t.phi == zero
    assert B.t.rho == zero

    pos_int = interior_samples(cavity.radius, cavity.length, ntests)
    ctpi4 = cavity.wavelength / 8

    def E(p: Cylindric3):
        E, _ = cavity._cylindric_field(p, ctpi4)
        return E

    divE = jax.vmap(DivergenceField(E))(pos_int.to_cylindric())
    assert divE == zero

    def B(p: Cylindric3):
        _, B = cavity._cylindric_field(p, ctpi4)
        return B

    divB = jax.vmap(DivergenceField(B))(pos_int.to_cylindric())
    assert divB == zero

    def Efield(p: Cartesian3):
        E, _ = cavity.field_strength(Cartesian4.make(x=p.x, y=p.y, z=p.z, ct=ctpi4))
        return E

    divE = jax.vmap(DivergenceField(Efield))(pos_int)

    def Bfield(p: Cartesian3):
        _, B = cavity.field_strength(Cartesian4.make(x=p.x, y=p.y, z=p.z, ct=ctpi4))
        return B

    divB = jax.vmap(DivergenceField(Bfield))(pos_int)
    assert divB == zero
