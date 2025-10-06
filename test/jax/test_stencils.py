import jax
import jax.numpy as jnp
import pytest

from beamline.jax.stencils import (
    prolongate1d,
    prolongate2d,
    prolongate3d,
    restrict1d,
    restrict2d,
    restrict3d,
)


@pytest.mark.parametrize("n", [3, 5, 9])
def test_multigrid(n):
    r1d = jax.lax.map(restrict1d, jnp.eye(n))
    assert jnp.allclose(r1d.sum(axis=1), 0.5)

    p1d = jax.lax.map(prolongate1d, jnp.eye(n))
    assert jnp.allclose(p1d.sum(axis=1), 2.0)

    rp1d = jax.lax.map(prolongate1d, r1d)
    assert jnp.allclose(rp1d.sum(axis=1), 1.0)

    r2d = jax.lax.map(restrict2d, jnp.eye(n * n).reshape(n * n, n, n))
    r2d_sum = r2d.sum(axis=(1, 2)).reshape(n, n)
    assert jnp.allclose(r2d_sum, 0.25)

    p2d = jax.lax.map(prolongate2d, jnp.eye(n * n).reshape(n * n, n, n))
    p2d_sum = p2d.sum(axis=(1, 2)).reshape(n, n)
    assert jnp.allclose(p2d_sum, 4.0)

    rp2d = jax.lax.map(prolongate2d, r2d)
    rp2d_sum = rp2d.sum(axis=(1, 2)).reshape(n, n)
    assert jnp.allclose(rp2d_sum, 1.0)

    r3d = jax.lax.map(restrict3d, jnp.eye(n * n * n).reshape(n * n * n, n, n, n))
    r3d_sum = r3d.sum(axis=(1, 2, 3)).reshape(n, n, n)
    assert jnp.allclose(r3d_sum, 0.125)

    p3d = jax.lax.map(prolongate3d, jnp.eye(n * n * n).reshape(n * n * n, n, n, n))
    p3d_sum = p3d.sum(axis=(1, 2, 3)).reshape(n, n, n)
    assert jnp.allclose(p3d_sum, 8.0)

    rp3d = jax.lax.map(prolongate3d, r3d)
    rp3d_sum = rp3d.sum(axis=(1, 2, 3)).reshape(n, n, n)
    assert jnp.allclose(rp3d_sum, 1.0)
