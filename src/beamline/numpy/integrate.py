"""Utilities for integratinge ODEs with pytree states."""

from collections.abc import Callable
from typing import TypeVar

import vector
from scipy.integrate import solve_ivp as _solve_ivp

pytree = vector.register_pytree()

T = TypeVar("T")
Fun = Callable[[float, T], T]
Event = Callable[[float, T], float]
Events = Event[T] | list[Event[T]] | None


class _FunWrapper:
    def __init__(self, unravel, fun: Fun[T]):
        self.unravel = unravel
        self.fun = fun

    def __call__(self, t, flat_y):
        y = self.unravel(flat_y)
        dy_dt = self.fun(t, y)
        flat_dy_dt, _ = pytree.ravel(dy_dt)
        return flat_dy_dt


class _EventWrapper:
    def __init__(self, unravel, event: Event[T]):
        self.unravel = unravel
        self.event = event

    def __call__(self, t, flat_y):
        y = self.unravel(flat_y)
        return self.event(t, y)

    @property
    def terminal(self):
        return getattr(self.event, "terminal", False)

    @property
    def direction(self):
        return getattr(self.event, "direction", 0)


def solve_ivp(
    fun: Fun[T],
    t_span: tuple[float, float],
    y0: T,
    *,
    events: Events[T] = None,
    **kwargs,
):
    r"""Integrate a system of ordinary differential equations.

    This is a wrapper around `scipy.integrate.solve_ivp` that allows the state to be
    an arbitrary pytree registered with `vector.register_pytree`.
    """
    flat_y0, unravel = pytree.ravel(y0)

    if events is not None:
        if isinstance(events, list):
            flat_event = [_EventWrapper(unravel, ev) for ev in events]
        else:
            flat_event = _EventWrapper(unravel, events)

        kwargs["events"] = flat_event

    solution = _solve_ivp(
        fun=_FunWrapper(unravel, fun),
        t_span=t_span,
        y0=flat_y0,
        **kwargs,
    )
    solution.y = [unravel(step) for step in solution.y.T]
    if solution.y_events is not None:
        solution.y_events = [
            [unravel(step) for step in event] for event in solution.y_events
        ]
    return solution
