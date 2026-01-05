"""Bessel functions

Needed mainly for the RF cavities

Unfortunately, JAX does not have built-in Bessel functions, other than
jax.scipy.special.bessel_jn, which has limited functionality.

A PR exists https://github.com/jax-ml/jax/pull/17038 to add Bessel functions to JAX, but it has not yet been merged.

A cosine-based series expansion can be found in https://arxiv.org/pdf/2206.05334
It is good for small x
"""