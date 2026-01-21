"""Helpers for JAX types

Useful type aliases, and common conventions in this module
"""

from jax import Array
from jaxtyping import Float, Int

SFloat = Float[Array, ""] | float
"""Scalar (double-precision) floating point"""
SInt = Int[Array, ""]
"""Scalar integer"""
