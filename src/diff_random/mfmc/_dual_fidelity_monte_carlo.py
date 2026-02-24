from abc import ABCMeta, abstractmethod
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp

type PRNGKeyArray = jax.Array
type StepSize = jax.Array
type LogWeight = jax.Array
type LogReweightRatio = jax.Array


class DualFidelityMonteCarlo[StateT: Any, ParamT: Any, LogWeightT: (LogWeight, None)](
    metaclass=ABCMeta
):
    """
    A class for creating dual-fidelity monte carlo simulators, for scenarios where
    doing multiple small steps with step-parameters computed using the same
    initial-state is statistically equivalent to performing one large step, with
    step size equal to the sum of the individual small step sizes.

    This class is intended to be subclassed to provide the following methods:
    (a) 'compute_step_param': computes the parameters of a simulation step
        based on the current state and step size.
    (b) 'single_step': computes the next state based on the current state, step size,
        step-parameters, and a random key. Optionally, 'single_step' should take
        alternative step-parameters as an additional argument and return the log of the
        reweight factor from the actual step-parameters to the alternative
        step-parameters.

    Using these:
    - The method called 'low_fidelity_step' performs a single step with
      step_size=lf_step_size. Step parameters are computed using 'compute_step_param'.
    - The method called 'high_fidelity_multistep' performs `num_hf_steps_per_lf_step`
      steps, each with step_size=lf_step_size/num_hf_steps_per_lf_step, and returns
      the state at the end of the "multistep". The parameters are computed using
      'compute_step_param' after each "inner-loop" step (this is what makes it high
      fidelity). Optionally, this method can also return a reweight factor to convert
      the high fidelity simulation into a low fidelity one. This may not sound useful,
      but is in the context of error estimation, multi fidelity MC estimation, etc.
    """

    def __init__(
        self,
        lf_step_size: StepSize,
        num_hf_steps_per_lf_step: int,
    ) -> None:
        self.lf_step_size = lf_step_size
        self.num_hf_steps_per_lf_step = num_hf_steps_per_lf_step

    def low_fidelity_step(
        self,
        init_state: StateT,
        key: PRNGKeyArray,
    ) -> tuple[StateT, LogWeightT]:
        step_param = self.compute_step_param(
            init_state=init_state,
            step_size=self.lf_step_size,
        )

        return self.single_step(
            init_state=init_state,
            step_size=self.lf_step_size,
            step_param=step_param,
            key=key,
            reweight_param=None,
        )[:2]

    @overload
    def high_fidelity_multistep(
        self,
        init_state: StateT,
        key: PRNGKeyArray,
        compute_lf_log_reweight: Literal[True] = True,
    ) -> tuple[StateT, LogWeightT, LogReweightRatio]: ...

    @overload
    def high_fidelity_multistep(
        self,
        init_state: StateT,
        key: PRNGKeyArray,
        compute_lf_log_reweight: Literal[False],
    ) -> tuple[StateT, LogWeightT, None]: ...

    def high_fidelity_multistep(
        self,
        init_state: StateT,
        key: PRNGKeyArray,
        compute_lf_log_reweight: bool = True,
        unroll: int | bool | None = None,
    ) -> tuple[StateT, LogWeightT, LogReweightRatio | None]:
        hf_step_size = self.lf_step_size / self.num_hf_steps_per_lf_step

        if compute_lf_log_reweight:
            lf_param_for_inner_loop = self.compute_step_param(
                init_state=init_state,
                step_size=hf_step_size,
            )
            init_log_reweight_ratio = jnp.array(1.0)
        else:
            lf_param_for_inner_loop = None
            init_log_reweight_ratio = None

        def high_fidelity_inner_step(
            idx: int,
            val: tuple[StateT, PRNGKeyArray, LogWeightT, LogReweightRatio | None],
        ) -> tuple[StateT, PRNGKeyArray, LogWeightT, LogReweightRatio | None]:
            (
                init_state,
                key,
                cumulative_log_weight,
                cumulative_log_reweight_ratio,
            ) = val

            hf_param = self.compute_step_param(
                init_state=init_state,
                step_size=hf_step_size,
            )

            cur_key, next_key = jax.random.split(key, num=2)

            (
                next_state,
                log_weight_update,
                log_reweight_ratio_update,
            ) = self.single_step(
                init_state=init_state,
                step_size=hf_step_size,
                step_param=hf_param,
                key=cur_key,
                reweight_param=lf_param_for_inner_loop,
            )

            if log_weight_update is not None:
                assert cumulative_log_weight is not None  # needed for ty
                cumulative_log_weight += (  # pyrefly: ignore[unsupported-operation]
                    log_weight_update
                )

            if log_reweight_ratio_update is not None:
                assert (
                    cumulative_log_reweight_ratio is not None
                )  # needed for all type checkers
                cumulative_log_reweight_ratio += log_reweight_ratio_update

            return (
                next_state,
                next_key,
                cumulative_log_weight,
                cumulative_log_reweight_ratio,
            )  # ty: ignore[invalid-return-type]

        return jax.lax.fori_loop(
            lower=0,
            upper=self.num_hf_steps_per_lf_step,
            body_fun=high_fidelity_inner_step,
            init_val=(init_state, key, jnp.array(1.0), init_log_reweight_ratio),
            unroll=unroll,
        )

    @staticmethod
    @abstractmethod
    def compute_step_param(
        init_state: StateT,
        step_size: StepSize,
    ) -> ParamT:
        raise NotImplementedError

    @overload
    @staticmethod
    def single_step(
        init_state: StateT,
        step_size: StepSize,
        step_param: ParamT,
        key: PRNGKeyArray,
        reweight_param: None = None,
    ) -> tuple[StateT, LogWeightT, None]: ...

    @overload
    @staticmethod
    def single_step(
        init_state: StateT,
        step_size: StepSize,
        step_param: ParamT,
        key: PRNGKeyArray,
        reweight_param: ParamT,
    ) -> tuple[StateT, LogWeightT, LogReweightRatio]: ...

    @staticmethod
    @abstractmethod
    def single_step(
        init_state: StateT,
        step_size: StepSize,
        step_param: ParamT,
        key: PRNGKeyArray,
        reweight_param: ParamT | None = None,
    ) -> tuple[StateT, LogWeightT, LogReweightRatio | None]:
        raise NotImplementedError
