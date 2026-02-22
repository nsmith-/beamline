import dataclasses
import math
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, dataclass_transform

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

# The different type aliases are primarily for documentation purposes

type ShapeLike = Sequence[int]

type DistParam = float | np.floating | np.ndarray | jnp.ndarray

type Sample = jax.Array
type LogWeight = jax.Array

# type BatchedSample = jax.Array
# type BatchedLogWeight = jax.Array
# type BatchedAuxInfo = jax.Array

# (sample, optional log-weight, optional aux-info)
# type GenResult = tuple[Sample, LogWeight | None, AuxInfo | None]
# type BatchedGenResult = \
#     tuple[BatchedSample, BatchedLogWeight | None, BatchedAuxInfo | None]
# type MaybeBatchedGenResult = GenResult | BatchedGenResult

# type Genresult_Simple = tuple[Sample, None, None]
# type GenResult_HasWeight = tuple[Sample, LogWeight, AuxInfo | None]
# type GenResult_HasAuxInfo = tuple[Sample, LogWeight | None, AuxInfo]
# type GenResult_HasWeightAuxInfo = tuple[Sample, LogWeight, AuxInfo]

type LogWeightRatio = jax.Array

type PRNGKeyArray = jax.Array


@dataclass_transform(
    eq_default=True,
    order_default=False,
    kw_only_default=False,
    frozen_default=True,
    field_specifiers=(dataclasses.field,),
)
class _DistributionBase[LogWeightT: (LogWeight, None), AuxT: Any, **DistParamSpec](
    metaclass=ABCMeta
):
    def __init_subclass__(cls, /, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(
            init=True,
            repr=True,
            eq=True,
            order=False,
            unsafe_hash=False,
            frozen=True,
            match_args=True,
            kw_only=False,
            slots=False,
            weakref_slot=False,
        )(cls)

    @abstractmethod
    def _generate_one_sample(
        self,
        key: PRNGKeyArray,  # traced, not diffnble wrt
        dtype: DTypeLike | None = None,  # not traced
    ) -> tuple[Sample, LogWeightT, AuxT]:
        raise NotImplementedError

    def reweight(
        self,
        gen_result: tuple[Sample, LogWeightT, AuxT],
        *new_args: DistParamSpec.args,
        **new_kwargs: DistParamSpec.kwargs,
    ) -> LogWeightRatio:
        raise NotImplementedError

    def generate(
        self,
        key: PRNGKeyArray,  # traced, not diffnble wrt
        batch_shape: ShapeLike | None = None,  # not traced
        dtype: DTypeLike | None = None,  # not traced
    ) -> tuple[Sample, LogWeightT, AuxT]:
        if batch_shape is None:
            return self._generate_one_sample(key, dtype)

        batch_shape = tuple(batch_shape)
        batch_size = math.prod(batch_shape)
        batched_key = jax.random.split(key, batch_size)

        samples, log_weights, aux_info = jax.vmap(
            self._generate_one_sample, in_axes=[0, None], out_axes=0
        )(batched_key, dtype)

        samples = samples.reshape(batch_shape + samples.shape[1:])

        if log_weights is not None:
            log_weights = log_weights.reshape(batch_shape + log_weights.shape[1:])

        if aux_info is not None:
            aux_info = aux_info.reshape(batch_shape + aux_info.shape[1:])

        return samples, log_weights, aux_info


# key = jax.random.key(0)
# N = 1_000_000

# def p_sq_estimate(p):
#     X, log_W = Bernoulli(p=p).generate(key, batch_shape=(N,))
#     p_estimate = jnp.sum(X * jnp.exp(log_W))/N
#     return p_estimate**2

# dbydp_of_p_sq = jax.grad(p_sq_estimate)
