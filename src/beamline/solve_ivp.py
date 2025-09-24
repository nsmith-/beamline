from typing import Callable, TypeVar
from scipy.integrate import solve_ivp
import vector

pytree = vector.register_pytree()

T = TypeVar("T")


def wrapped_solve(fun: Callable[[float, T], T], y0: T, **kwargs):
    flat_y0, unravel = pytree.ravel(y0)

    def wrap_event(event: Callable[[float, T], float]):
        def flat_event(t, flat_y):
            state = unravel(flat_y)
            return event(t, state)

        flat_event.terminal = getattr(event, "terminal", False)
        flat_event.direction = getattr(event, "direction", 0)
        return flat_event

    if "events" in kwargs:
        if isinstance(kwargs["events"], list):
            flat_event = [wrap_event(ev) for ev in kwargs["events"]]
        else:
            flat_event = wrap_event(kwargs["events"])

        kwargs["events"] = flat_event

    def flat_fun(t, flat_y):
        state = unravel(flat_y)
        dstate_dt = fun(t, state)
        flat_dstate_dt, _ = pytree.ravel(dstate_dt)
        return flat_dstate_dt

    solution = solve_ivp(
        fun=flat_fun,
        y0=flat_y0,
        **kwargs,
    )
    solution.y = [unravel(step) for step in solution.y.T]
    if solution.y_events is not None:
        solution.y_events = [
            [unravel(step) for step in event] for event in solution.y_events
        ]
    return solution
