from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ._distribution_base import (
    DistParam,
    LogWeight,
    LogWeightRatio,
    PRNGKeyArray,
    Sample,
    _DistributionBase,
)


## log-weight for Bernoulli samples, with a custom gradient ####################
@jax.custom_jvp
def _bernoulli_log_w_func(p: DistParam, x: Sample) -> LogWeight:
    assert jnp.shape(p) == jnp.shape(x)
    return jnp.zeros_like(p)


# The weight gradients will be nan if p = 0. or p = 1.
# This behavior is intentional (can be changed easily, but shouldn't).
@_bernoulli_log_w_func.defjvp
def _jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, Any],
) -> tuple[jax.Array, jax.Array]:
    p, x = primals
    p_dot, x_dot = tangents

    assert jnp.shape(p) == jnp.shape(x)
    del x_dot

    primal_out = jnp.zeros_like(p)

    tangent_out = jnp.where(x, p_dot / p, p_dot / (p - 1))

    # This version will make the log_weight gradients finite at p = 0. and p = 1.:
    # tangent_out = p_dot / jnp.where(x, p, p-1)

    return primal_out, tangent_out


del _jvp

## Functionally (but not operationally) equivalent definition ########
# def _bernoulli_log_w_func(p: DistParam, x: Sample) -> LogWeight:
#     ## With nan grad_log_w at p = 0. and p = 1.
#     p_no_grad = jax.lax.stop_gradient(p)
#     return jnp.where(
#         x,
#         jnp.log(p) - jnp.log(p_no_grad),
#         jnp.log(1-p) - jnp.log(1-p_no_grad)
#     )

#     ## Without nan grad_log_w at p = 0. and p = 1.
#     # tmp = jnp.where(x, p, (1-p))
#     # return jnp.log(tmp) - jnp.log(jax.lax.stop_gradient(tmp))

######################################################################
################################################################################


class Bernoulli(_DistributionBase[LogWeight, None, [DistParam]]):
    p: DistParam = 0.5

    def _generate_one_sample(
        self,
        key: PRNGKeyArray,
        dtype: DTypeLike | None = None,
    ) -> tuple[Sample, LogWeight, None]:
        p = self.p
        p_no_grad = jax.lax.stop_gradient(p)

        x = jax.random.bernoulli(
            key=key,
            p=p_no_grad,
            shape=None,
            mode="low",
        )

        log_w = _bernoulli_log_w_func(p, x)

        return x, log_w, None

    def reweight(
        self,
        gen_result: tuple[Sample, Any, Any],
        new_p: DistParam,
    ) -> LogWeightRatio:
        x, _, _ = gen_result
        orig_p = self.p

        return jnp.where(
            x,
            (jnp.log(new_p) - jnp.log(orig_p)),
            (jnp.log(1 - new_p) - jnp.log(1 - orig_p)),
        )
