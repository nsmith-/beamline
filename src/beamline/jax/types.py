"""Helpers for JAX types

Useful type aliases, and common conventions in this module
"""

from jax import Array
from jaxtyping import Float, Int

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
