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
    ShapeLike,
    _DistributionBase,
)

type LandauAuxInfo = jax.Array


class _MultichannelLandauBase[LogWeightT: (LogWeight, None)](
    _DistributionBase[LogWeightT, LandauAuxInfo, [DistParam, DistParam], int]
):
    loc: DistParam  # traced, diffnble wrt
    scale: DistParam  # traced, diffnble wrt

    def generate(  # type: ignore[override]
        self,
        key: PRNGKeyArray,  # traced, not diffnble wrt
        batch_shape: ShapeLike | None = None,  # not traced
        dtype: DTypeLike | None = None,  # not traced
        num_channels: int = 1,
    ) -> tuple[
        Sample, LogWeightT, LandauAuxInfo
    ]:  # ty: ignore[invalid-method-override]
        return super().generate(  # type: ignore[return-value]
            key=key,
            batch_shape=batch_shape,
            dtype=dtype,
            extra_arg=num_channels,
        )

    @staticmethod
    def _generate_standard_landau(
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
        num_channels: int = 1,
    ) -> tuple[Sample, LandauAuxInfo]:
        unif_key, exp_key, choice_key = jax.random.split(key, num=3)
        pi_by_2 = jnp.pi / 2

        # Initial experiments suggest that option 1 is better than option 2, and so on.
        ## Option 1: Low discrepancy sequence w/ random first value #######
        alpha = (jnp.sqrt(5) - 1) / 2
        start = jax.random.uniform(
            key=unif_key,
            shape=(*shape, 1),
            dtype=dtype,
            minval=0,
            maxval=1,
        )
        unif_rv_cand = -pi_by_2 + jnp.pi * (
            (start + alpha * jnp.arange(num_channels)) % 1
        )
        del alpha, start
        ###################################################################

        ## Option 2: IID sampling #########################################
        # unif_rv_cand = jax.random.uniform(
        #     key=unif_key,
        #     shape=(*shape, num_channels),
        #     dtype=dtype,
        #     minval=-pi_by_2,
        #     maxval=pi_by_2,
        # )
        ###################################################################

        ## Option 3: Equidistant points with random offset ################
        # width = jnp.pi / num_channels
        # start = jnp.linspace(-pi_by_2, pi_by_2, num_channels, endpoint=False)
        # unif_rv_cand = start + width * jax.random.uniform(
        #     key=unif_key,
        #     shape=(*shape, 1),
        #     dtype=dtype,
        #     minval=0,
        #     maxval=1,
        # )
        # del width, start
        ###################################################################

        ## Option 4: Independent samples from each quantile ###############
        # width = jnp.pi / num_channels
        # start = jnp.linspace(-pi_by_2, pi_by_2, num_channels, endpoint=False)
        # unif_rv_cand = start + width * jax.random.uniform(
        #     key=unif_key,
        #     shape=(*shape, num_channels),
        #     dtype=dtype,
        #     minval=0,
        #     maxval=1,
        # )
        # del width, start
        ###################################################################

        exp_rv = jax.random.exponential(
            key=exp_key,
            shape=shape,
            dtype=dtype,
        )

        del unif_key, exp_key

        aux_info = (1 / pi_by_2) * (
            (pi_by_2 + unif_rv_cand) * jnp.tan(unif_rv_cand)
            - jnp.log((pi_by_2 * jnp.cos(unif_rv_cand)) / (pi_by_2 + unif_rv_cand))
        )

        aux_info_chosen = jax.random.choice(key=choice_key, a=aux_info, axis=-1)

        std_landau_rv = aux_info_chosen - jnp.log(exp_rv) / pi_by_2

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
        #         = (p(aux_info) * pi_by_2 / scale) * (1/K) * sum_k exp_rv_cand[k] * p(exp_rv_cand[k])
        # w_ratio = new_p / orig_p
        #         = (sum_k new_exp_rv_cand[k] exp(-new_exp_rv_cand[k]))
        #           / (sum_k orig_exp_rv_cand[k] exp(-orig_exp_rv_cand[k]))
        #           * (orig_scale / new_scale)
        # log_w_ratio = (orig_std_landau_rv - new_std_landau_rv) * pi_by_2
        #               + log(sum_k exp(aux_info[k] * pi_by_2 - new_exp_rv_cand[k]))
        #               - log(sum_k exp(aux_info[k] * pi_by_2 - orig_exp_rv_cand[k]))
        #               + log(orig_scale) - log(new_scale)

        landau_rv, _, aux_info = gen_result
        assert landau_rv is not None
        assert aux_info is not None

        orig_loc = self.loc
        orig_scale = self.scale

        orig_std_landau_rv = (landau_rv - orig_loc) / orig_scale
        new_std_landau_rv = (landau_rv - new_loc) / new_scale

        pi_by_2 = jnp.pi / 2

        orig_exp_rv_candidates = jnp.exp(
            (aux_info - orig_std_landau_rv[..., None]) * pi_by_2
        )
        new_exp_rv_candidates = jnp.exp(
            (aux_info - new_std_landau_rv[..., None]) * pi_by_2
        )

        log_w_ratio = (
            (orig_std_landau_rv - new_std_landau_rv) * pi_by_2
            + jax.scipy.special.logsumexp(
                aux_info * pi_by_2 - new_exp_rv_candidates, axis=-1
            )
            - jax.scipy.special.logsumexp(
                aux_info * pi_by_2 - orig_exp_rv_candidates, axis=-1
            )
            + (jnp.log(orig_scale) - jnp.log(new_scale))
        )

        return log_w_ratio


class MultichannelLandau_SG(_MultichannelLandauBase[None]):
    def _generate_one_sample(  # ty: ignore[invalid-method-override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        key: PRNGKeyArray,
        dtype: DTypeLike | None = None,
        num_channels: int = 1,  # type: ignore[override]
    ) -> tuple[Sample, None, LandauAuxInfo]:
        single_sample_shape = jnp.broadcast_shapes(
            jnp.shape(self.loc),
            jnp.shape(self.scale),
        )

        std_landau_rv, aux_info = self._generate_standard_landau(
            key=key,
            shape=single_sample_shape,
            dtype=dtype,
            num_channels=num_channels,
        )

        landau_rv = (std_landau_rv * self.scale) + self.loc

        return landau_rv, None, aux_info


class MultichannelLandau_WG(_MultichannelLandauBase[LogWeight]):
    def _generate_one_sample(  # ty: ignore[invalid-method-override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        key: PRNGKeyArray,
        dtype: DTypeLike | None = None,
        num_channels: int = 1,  # type: ignore[override]
    ) -> tuple[Sample, LogWeight, LandauAuxInfo]:
        single_sample_shape = jnp.broadcast_shapes(
            jnp.shape(self.loc),
            jnp.shape(self.scale),
        )

        std_landau_rv, aux_info = self._generate_standard_landau(
            key=key,
            shape=single_sample_shape,
            dtype=dtype,
            num_channels=num_channels,
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
