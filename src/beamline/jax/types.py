"""Helpers for JAX types

Useful type aliases, and common conventions in this module

We try to quarantine all jaxtyping declarations here so we can
disable the linter error about annotations just for this file.
"""

from jax import Array
from jaxtyping import Float, Int

# TODO: | float is a crutch.. need to get it out
SFloat = Float[Array, ""] | float
"""Scalar (double-precision) floating point"""
SInt = Int[Array, ""]
"""Scalar integer"""
VecN = Float[Array, "N"]
"""N-dimensional vector of floating point numbers"""
Vec3 = Float[Array, "3"]
"""3D vector of floating point numbers"""
Vec4 = Float[Array, "4"]
"""4D vector of floating point numbers"""
Rot4Matrix = Float[Array, "4 4"]
"""4x4 rotation matrix (Lorentz transformation)"""
