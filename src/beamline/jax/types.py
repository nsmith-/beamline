"""Helpers for JAX types

Useful type aliases, and common conventions in this module

We try to quarantine all jaxtyping declarations here so we can
disable the linter error about annotations just for this file.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

# TODO: | float is a crutch.. need to get it out?
# similar situation to diffrax RealScalarLike, use it?
SFloat = Float[Array, ""] | float
"""Scalar (double-precision) floating point"""
SInt = Int[Array, ""] | int
"""Scalar integer"""
SBool = Bool[Array, ""] | bool
"""Scalar boolean"""
VecN = Float[Array, "N"]
"""N-dimensional vector of floating point numbers"""
Vec3 = Float[Array, "3"]
"""3D vector of floating point numbers"""
Vec4 = Float[Array, "4"]
"""4D vector of floating point numbers"""
Rot4Matrix = Float[Array, "4 4"]
"""4x4 rotation matrix (Lorentz transformation)"""


def dtype_of(x: SFloat | SInt | SBool) -> jnp.dtype:
    """Get the dtype of a scalar type alias"""
    if isinstance(x, float):
        return jnp.dtype(jnp.float64)
    elif isinstance(x, int):
        return jnp.dtype(jnp.int64)
    elif isinstance(x, bool):
        return jnp.dtype(jnp.bool)
    return x.dtype


def eps_of(x: SFloat) -> SFloat:
    """Get the machine epsilon for the dtype of a scalar type alias"""
    return jnp.finfo(dtype_of(x)).eps
