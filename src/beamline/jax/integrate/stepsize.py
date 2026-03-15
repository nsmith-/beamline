from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from diffrax import AbstractStepSizeController, AbstractTerm
from jaxtyping import PyTree

from beamline.jax.types import SBool, SFloat, SInt, eps_of


class BoundaryAwareStepSizeController[ControllerState, DT0: SFloat | None, Y](
    AbstractStepSizeController[ControllerState, DT0]
):
    """A boundary-aware step size controller

    Args:
        controller: The underlying step size controller to use
        sdf: A signed distance function that gives the distance
            to the nearest boundary. If the distance is negative,
            the closest boundary is behind the current position.
    """

    controller: AbstractStepSizeController[ControllerState, DT0]
    sdf: Callable[[Y], SFloat]

    def __init__(
        self,
        controller: AbstractStepSizeController[ControllerState, DT0],
        sdf: Callable[[Y], SFloat],
    ):
        self.controller = controller
        self.sdf = sdf

    def wrap(self, direction: SInt):
        controller = self.controller.wrap(direction)
        # do we care about the direction? abs(signed distance) is a good max step
        return eqx.tree_at(
            lambda s: (s.controller,),
            self,
            (controller,),
            is_leaf=lambda x: x is None,
        )

    def init[Args](
        self,
        terms: PyTree[AbstractTerm],
        t0: SFloat,
        t1: SFloat,
        y0: Y,
        dt0: DT0,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], SFloat, Y, Args], Any],
        error_order: SFloat | None,
    ) -> tuple[SFloat, ControllerState]:
        sdf = self.sdf(y0)
        # TODO: assert > 0, don't start at a boundary!
        dt0 = jnp.minimum(dt0, jnp.abs(sdf))
        # TODO: create a state wrapper that tracks the last sdf
        # we can use it to detect overshooting the boundary and possibly reject a step
        # this also lets us signal the vector field discontinuity has been passed
        return self.controller.init(terms, t0, t1, y0, dt0, args, func, error_order)

    def adapt_step_size(
        self,
        t0: SFloat,
        t1: SFloat,
        y0: Y,
        y1_candidate: Y,
        args: Any,
        y_error: Y | None,
        error_order: SFloat,
        controller_state: ControllerState,
    ) -> tuple[
        SBool,
        SFloat,
        SFloat,
        SBool,
        ControllerState,
        Any,
    ]:
        (
            keep_step,
            next_t0,
            next_t1,
            made_jump,
            state,
            result,
        ) = self.controller.adapt_step_size(
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            y_error,
            error_order,
            controller_state,
        )
        sdf = self.sdf(y1_candidate)
        max_next_step = jnp.where(
            sdf <= 0,
            jnp.abs(sdf) + eps_of(sdf),
            sdf,
        )
        next_t1 = jnp.minimum(next_t1, next_t0 + max_next_step)
        return keep_step, next_t0, next_t1, made_jump, state, result
