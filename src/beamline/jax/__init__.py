"""Jax-based implementations

This module sets up JAX to use 64-bit floating point precision by default.
"""
import jax

jax.config.update("jax_enable_x64", True)