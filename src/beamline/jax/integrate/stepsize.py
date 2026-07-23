from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractStepSizeController, AbstractTerm
from jaxtyping import PyTree

from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat, SInt, eps_of

type _BoundaryState[ControllerState] = tuple[SFloat, SFloat, ControllerState]
"""A state wrapper for the boundary-aware step size controller

The first element is the last signed distance function value, which can be used
to detect when a boundary has been crossed.

The second element is the initial step size, which steers how to restart the
inner controller after crossing a boundary.

The third element is the state of the underlying controller.
"""


class BoundaryAwareStepSizeController[ControllerState, DT0: SFloat, Y: ParticleState](
    AbstractStepSizeController[_BoundaryState[ControllerState], DT0]
):
    """A boundary-aware step size controller

    Args:
        controller: The underlying step size controller to use
        sdf: A signed time to boundary function (``Volume.signed_time_to_boundary``
            evaluated on the current state). Positive when outside the volume
            (value = time to entry), negative when inside (magnitude = time to
            exit). The absolute value is used as the step size limit.
        max_step: A maximum step size to prevent the solver from
            completely jumping over a volume.
        crossing_eps: The desired step size at which we cross a boundary,
            relative to the current time. The sdf is used to approach the
            boundary up to -crossing_eps, and then the next step will be
            2*crossing_eps. Defaults to 10x the precision of the time step
            if not specified.
    """

    controller: AbstractStepSizeController[ControllerState, DT0]
    sdf: Callable[[Y], SFloat]
    max_step: SFloat
    crossing_eps: SFloat | None
    debug: bool

    def __init__(
        self,
        controller: AbstractStepSizeController[ControllerState, DT0],
        sdf: Callable[[Y], SFloat],
        max_step: SFloat,
        *,
        crossing_eps: SFloat | None = None,
        debug: bool = False,
    ):
        self.controller = controller
        self.sdf = sdf
        self.max_step = max_step
        self.crossing_eps = crossing_eps
        self.debug = debug

    def wrap(self, direction: SInt):
        controller = self.controller.wrap(direction)

        def sdf(y: Y):
            # scale tangent vector by the integration direction
            y = eqx.tree_at(lambda s: s.kin, y, replace=y.kin * direction)
            return self.sdf(y)

        return eqx.tree_at(
            lambda s: (s.controller, s.sdf),
            self,
            (controller, sdf),
            is_leaf=lambda x: x is None,
        )

    def _crossing_eps(self, val) -> SFloat:
        """Return the effective crossing epsilon for the given value"""
        if self.crossing_eps is None:
            return (10 * eps_of(val)) * jnp.abs(val)
        return self.crossing_eps * jnp.abs(val)

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
        max_next_step = self._softclip_sdf(sdf)
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
        # signs of crossing a boundary: sdf goes from positive to negative or vice versa
        # or the sdf goes to infinity (exit the volume and no future intersection)
        # TODO: reject step if the jump went too far past the boundary?
        made_jump = (
            made_jump
            | (last_sdf * sdf <= 0)
            | (jnp.isfinite(last_sdf) & ~jnp.isfinite(sdf))
        )
        # get an epsilon at the correct scale for the addition
        crossing_eps = self._crossing_eps(next_t0)
        next_t1 = jnp.where(
            made_jump,
            # use initial step size since we made a jump
            next_t0 + dt0,
            jnp.where(
                jnp.abs(sdf) <= crossing_eps,
                # we are about to cross the boundary, so take a special step to cross it
                next_t0 + 2 * crossing_eps,
                # the next boundary is at least abs(sdf) away
                jnp.minimum(
                    old_next_t1,
                    next_t0 + self._softclip_sdf(jnp.abs(sdf)) - crossing_eps / 2,
                ),
            ),
        )

        if self.debug:
            jax.debug.print(
                "next_t0={next_t0} old_next_t1={old_next_t1}\n"
                "sdf={sdf} made_jump={made_jump} crossing_eps={crossing_eps}\n"
                "next_t1={next_t1}\n",
                next_t0=next_t0,
                old_next_t1=old_next_t1,
                sdf=sdf,
                made_jump=made_jump,
                crossing_eps=self._crossing_eps(next_t0),
                next_t1=next_t1,
            )
        state = (sdf, dt0, inner_state)
        return keep_step, next_t0, next_t1, made_jump, state, result
