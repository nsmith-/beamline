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

type LandauAuxInfo = jax.Array


class _LandauBase[LogWeightT: (LogWeight, None)](
    _DistributionBase[LogWeightT, LandauAuxInfo, [DistParam, DistParam]]
):
    loc: DistParam  # traced, diffnble wrt
    scale: DistParam  # traced, diffnble wrt

    @staticmethod
    def _generate_standard_landau(
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> tuple[Sample, LandauAuxInfo]:
        unif_key, exp_key = jax.random.split(key)
        pi_by_2 = jnp.pi / 2

        unif_rv = jax.random.uniform(
            key=unif_key,
            shape=shape,
            dtype=dtype,
            minval=-pi_by_2,
            maxval=pi_by_2,
        )

        exp_rv = jax.random.exponential(
            key=exp_key,
            shape=shape,
            dtype=dtype,
        )

        del unif_key, exp_key

        aux_info = (1 / pi_by_2) * (
            (pi_by_2 + unif_rv) * jnp.tan(unif_rv)
            - jnp.log((pi_by_2 * jnp.cos(unif_rv)) / (pi_by_2 + unif_rv))
        )

        std_landau_rv = aux_info - jnp.log(exp_rv) / pi_by_2

        # landau_rv = (std_landau_rv * scale) + loc
        # Gradients wrt loc and scale are handled in different ways
        # in the subclasses of `_LandauBase`.

        return std_landau_rv, aux_info

    def reweight(
        self,
        gen_result: tuple[Sample, Any, LandauAuxInfo],
        new_loc: DistParam,
        new_scale: DistParam,
    ) -> LogWeightRatio:
        # p(landau_rv, aux_info ; loc, scale)
        #         = p(aux_info) * pi_by_2 * exp_rv * p(exp_rv) / scale
        # w_ratio = new_p / orig_p
        #         = (new_exp_rv / orig_exp_rv)
        #           * p(new_exp_rv) / p(orig_exp_rv)
        #           * (orig_scale / new_scale)
        # log_w_ratio = log(new_exp_rv) - log(orig_exp_rv)
        #               + (orig_exp_rv - new_exp_rv)
        #               + log(orig_scale) - log(new_scale)

        landau_rv, _, aux_info = gen_result
        assert landau_rv is not None
        assert aux_info is not None

        orig_loc = self.loc
        orig_scale = self.scale

        orig_std_landau_rv = (landau_rv - orig_loc) / orig_scale
        new_std_landau_rv = (landau_rv - new_loc) / new_scale

        pi_by_2 = jnp.pi / 2

        orig_exp_rv = jnp.exp((aux_info - orig_std_landau_rv) * pi_by_2)
        new_exp_rv = jnp.exp((aux_info - new_std_landau_rv) * pi_by_2)

        log_w_ratio = (
            (orig_std_landau_rv - new_std_landau_rv) * pi_by_2
            + (orig_exp_rv - new_exp_rv)
            + (jnp.log(orig_scale) - jnp.log(new_scale))
        )

        return log_w_ratio


class Landau_SG(_LandauBase[None]):
    def _generate_one_sample(
        self,
        key: PRNGKeyArray,
        dtype: DTypeLike | None = None,
    ) -> tuple[Sample, None, LandauAuxInfo]:
        single_sample_shape = jnp.broadcast_shapes(
            jnp.shape(self.loc),
            jnp.shape(self.scale),
        )

        std_landau_rv, aux_info = self._generate_standard_landau(
            key=key,
            shape=single_sample_shape,
            dtype=dtype,
        )

        landau_rv = (std_landau_rv * self.scale) + self.loc

        return landau_rv, None, aux_info


class Landau_WG(_LandauBase[LogWeight]):
    def _generate_one_sample(
        self,
        key: PRNGKeyArray,
        dtype: DTypeLike | None = None,
    ) -> tuple[Sample, LogWeight, LandauAuxInfo]:
        single_sample_shape = jnp.broadcast_shapes(
            jnp.shape(self.loc),
            jnp.shape(self.scale),
        )

        std_landau_rv, aux_info = self._generate_standard_landau(
            key=key,
            shape=single_sample_shape,
            dtype=dtype,
        )

        loc_no_grad = jax.lax.stop_gradient(self.loc)
        scale_no_grad = jax.lax.stop_gradient(self.scale)

        landau_rv = (std_landau_rv * scale_no_grad) + loc_no_grad

        # - Some calculations within reweight() are redundant. They can be
        #   eliminated, possibly by implementing a custom gradient.
        # - The minus sign is because we want to reweight to (loc, scale) from
        #   (loc_no_grad, scale_no_grad) and not the other way around.
        log_w = -self.reweight(
            gen_result=(landau_rv, None, aux_info),
            new_loc=loc_no_grad,
            new_scale=scale_no_grad,
        )

        return landau_rv, log_w, aux_info
