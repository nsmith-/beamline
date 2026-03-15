from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractStepSizeController, AbstractTerm
from jaxtyping import PyTree

from beamline.jax.types import SBool, SFloat, SInt, eps_of

type _BoundaryState[ControllerState] = tuple[SFloat, SFloat, ControllerState]
"""A state wrapper for the boundary-aware step size controller

The first element is the last signed distance function value, which can be used
to detect when a boundary has been crossed.

The second element is the initial step size, which steers how to restart the
inner controller after crossing a boundary.

The third element is the state of the underlying controller.
"""


class BoundaryAwareStepSizeController[ControllerState, DT0: SFloat, Y](
    AbstractStepSizeController[_BoundaryState[ControllerState], DT0]
):
    """A boundary-aware step size controller

    Args:
        controller: The underlying step size controller to use
        sdf: A signed distance function that gives the distance
            to the nearest boundary. If the distance is negative,
            the closest boundary is behind the current position.
        max_step: A maximum step size to prevent the solver from
            completely jumping over a volume.
    """

    controller: AbstractStepSizeController[ControllerState, DT0]
    sdf: Callable[[Y], SFloat]
    max_step: SFloat

    def __init__(
        self,
        controller: AbstractStepSizeController[ControllerState, DT0],
        sdf: Callable[[Y], SFloat],
        max_step: SFloat,
    ):
        self.controller = controller
        self.sdf = sdf
        self.max_step = max_step

    def wrap(self, direction: SInt):
        controller = self.controller.wrap(direction)

        # TODO: direction < 0 not validated!
        def sdf(y):
            return self.sdf(y) * direction

        return eqx.tree_at(
            lambda s: (s.controller, s.sdf),
            self,
            (controller, sdf),
            is_leaf=lambda x: x is None,
        )

    def _softclip_sdf(self, sdf: SFloat) -> SFloat:
        """Clip large distances

        As sdf gets close to 0, this leaves it alone, but at large
        values it clips it to never exceed self.max_step. This makes
        the step size controller a bit more conservative about large jumps,
        to prevent it from completely jumping over a volume.
        """
        # pass inf through, but clip large finite values
        return jnp.where(
            jnp.isfinite(sdf),
            self.max_step * sdf / jnp.hypot(self.max_step, sdf),
            sdf,
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
    ) -> tuple[SFloat, _BoundaryState[ControllerState]]:
        t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        sdf = self.sdf(y0)
        max_next_step = jnp.where(
            sdf <= 0,
            # we just passed a boundary, the inner controller will probably return dt0
            jnp.inf,
            self._softclip_sdf(sdf),
        )
        jax.debug.print("initial sdf: {sdf}", sdf=sdf)
        t1 = jnp.minimum(t1, t0 + max_next_step)
        return t1, (sdf, dt0, inner_state)

    def adapt_step_size(
        self,
        t0: SFloat,
        t1: SFloat,
        y0: Y,
        y1_candidate: Y,
        args: Any,
        y_error: Y | None,
        error_order: SFloat,
        controller_state: _BoundaryState[ControllerState],
    ) -> tuple[
        SBool,
        SFloat,
        SFloat,
        SBool,
        _BoundaryState[ControllerState],
        Any,
    ]:
        last_sdf, dt0, inner_state = controller_state
        (
            keep_step,
            next_t0,
            old_next_t1,
            made_jump,
            inner_state,
            result,
        ) = self.controller.adapt_step_size(
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            y_error,
            error_order,
            inner_state,
        )
        sdf = self.sdf(y1_candidate)
        # TODO: reject step if the jump went too far past the boundary?
        made_jump = made_jump | ((last_sdf > 0) & (sdf <= 0))
        next_t1 = jnp.where(
            made_jump,
            next_t0 + dt0,
            jnp.where(
                jnp.abs(sdf) <= eps_of(sdf) * jnp.abs(next_t0),
                next_t0 + 2 * eps_of(sdf) * jnp.abs(next_t0),
                # the next boundary is at least abs(sdf) away
                jnp.minimum(old_next_t1, next_t0 + self._softclip_sdf(jnp.abs(sdf))),
            ),
        )

        if False:
            jax.debug.print(
                # "y1 candidate z {z} ct {ct} beta {beta}\n"
                "sdf at y1c: {sdf} old next t0: {next_t0} old next t1: {old_next_t1}\n"
                "new next t1: {calculated_next_t1}\n"
                "keep: {keep_step} made_jump: {made_jump}\n",
                # z=y1_candidate.kin.p.z,
                # ct=y1_candidate.kin.t.ct,
                # beta=y1_candidate.beta(),
                sdf=sdf,
                next_t0=next_t0,
                old_next_t1=old_next_t1,
                calculated_next_t1=next_t1,
                keep_step=keep_step,
                made_jump=made_jump,
            )
        state = (sdf, dt0, inner_state)
        return keep_step, next_t0, next_t1, made_jump, state, result
