"""Elliptic integrals

References:
https://github.com/sinaatalay/jaxellip
https://github.com/tagordon/ellip
https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/ellint_rf.hpp
https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/ellint_rd.hpp
https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/ellint_rj.hpp

Carlson "Numerical computation of real or complex elliptic integrals"
https://arxiv.org/abs/math/9409227
"""

from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax

_RFState: TypeAlias = tuple[Array, Array, Array, Array, Array, Array]


def _elliprf_full_iter(x: Array, y: Array, z: Array) -> Array:
    """Iterative solution for non-special cases"""

    def cond_fun(state: _RFState):
        _, _, _, A, Q, _ = state
        return Q >= jnp.abs(A)

    def body_fun(state: _RFState) -> _RFState:
        x, y, z, A, Q, f = state
        sqrtx = jnp.sqrt(x)
        sqrty = jnp.sqrt(y)
        sqrtz = jnp.sqrt(z)
        lam = sqrtx * sqrty + sqrtx * sqrtz + sqrty * sqrtz
        x_new = (x + lam) / 4
        y_new = (y + lam) / 4
        z_new = (z + lam) / 4
        A_new = (A + lam) / 4
        Q_new = Q / 4
        f_new = f * 4
        return x_new, y_new, z_new, A_new, Q_new, f_new

    A0 = (x + y + z) / 3
    Q = jnp.pow(3 * jnp.finfo(x.dtype).eps, -1 / 8) * jnp.max(
        jnp.array([jnp.abs(A0 - x), jnp.abs(A0 - y), jnp.abs(A0 - z)])
    )
    f = jnp.ones_like(x)

    _, _, _, A_final, _, f_final = lax.while_loop(
        cond_fun, body_fun, (x, y, z, A0, Q, f)
    )

    X = (A0 - x) / (A_final * f_final)
    Y = (A0 - y) / (A_final * f_final)
    Z = -(X + Y)

    E2 = X * Y - Z * Z
    E3 = X * Y * Z

    return (
        1.0
        + E3 * (1.0 / 14 + 3 * E3 / 104)
        + E2 * (-1.0 / 10 + E2 / 24 - (3 * E3) / 44 - 5 * E2 * E2 / 208 + E2 * E3 / 16)
    ) / jnp.sqrt(A_final)


_RF0State: TypeAlias = tuple[Array, Array]


def _elliprf_one_zero_iter(x: Array, y: Array) -> Array:
    """Solution for the case where one argument is zero"""
    xn = jnp.sqrt(x)
    yn = jnp.sqrt(y)

    def cond_fn(state: _RF0State):
        x, y = state
        return jnp.abs(x - y) >= 2.7 * jnp.finfo(x.dtype).eps * jnp.abs(x)

    def body_fn(state: _RF0State) -> _RF0State:
        x, y = state
        t = jnp.sqrt(x * y)
        x_new = (x + y) / 2
        y_new = t
        return x_new, y_new

    x_final, y_final = lax.while_loop(cond_fn, body_fn, (xn, yn))
    return jnp.pi / (x_final + y_final)


def elliprf(x: Array, y: Array, z: Array) -> Array:
    r"""Carlson symmetric elliptic integral of the first kind

    $$ R_F(x, y, z) = \frac{1}{2} \int_0^\infty \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}} $$

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z >= 0)

    At most one of x, y, z can be zero.

    Returns:
        R_F(x, y, z)
    """
    # TODO: there are additional special cases, is it worth implementing them?
    # See Boost implementation for reference
    lo, me, hi = jnp.sort(jnp.array([x, y, z]))
    return lax.select(
        lo == 0.0,
        _elliprf_one_zero_iter(me, hi),
        _elliprf_full_iter(lo, me, hi),
    )


_RDState: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


def _elliprd_general_iter(x: Array, y: Array, z: Array) -> Array:
    """General iterative solution for RD"""

    def cond_fun(state: _RDState):
        _, _, _, A, Q, _, _ = state
        return Q >= A

    def body_fun(state: _RDState) -> _RDState:
        x, y, z, A, Q, sum_term, f = state
        sqrtx = jnp.sqrt(x)
        sqrty = jnp.sqrt(y)
        sqrtz = jnp.sqrt(z)
        lam = sqrtx * sqrty + sqrtx * sqrtz + sqrty * sqrtz
        sum_term_new = sum_term + f / (sqrtz * (z + lam))
        x_new = (x + lam) / 4
        y_new = (y + lam) / 4
        z_new = (z + lam) / 4
        A_new = (A + lam) / 4
        Q_new = Q / 4
        f_new = f / 4
        return x_new, y_new, z_new, A_new, Q_new, sum_term_new, f_new

    A0 = (x + y + 3 * z) / 5
    Q = (
        jnp.pow(jnp.finfo(x).eps / 4, -1 / 8)
        * jnp.max(jnp.array([jnp.abs(A0 - x), jnp.abs(A0 - y), jnp.abs(A0 - z)]))
        * 1.2
    )
    f = jnp.ones_like(x)
    sum_term = jnp.zeros_like(x)

    _, _, _, A_final, _, sum_final, f_final = lax.while_loop(
        cond_fun, body_fun, (x, y, z, A0, Q, sum_term, f)
    )

    X = f_final * (A0 - x) / A_final
    Y = f_final * (A0 - y) / A_final
    Z = -(X + Y) / 3

    E2 = X * Y - 6 * Z * Z
    E3 = (3 * X * Y - 8 * Z * Z) * Z
    E4 = 3 * (X * Y - Z * Z) * Z * Z
    E5 = X * Y * Z * Z * Z

    result = (
        f_final
        * jnp.pow(A_final, -3 / 2)
        * (
            1
            - 3 * E2 / 14
            + E3 / 6
            + 9 * E2 * E2 / 88
            - 3 * E4 / 22
            - 9 * E2 * E3 / 52
            + 3 * E5 / 26
            - E2 * E2 * E2 / 16
            + 3 * E3 * E3 / 40
            + 3 * E2 * E4 / 20
            + 45 * E2 * E2 * E3 / 272
            - 9 * (E3 * E4 + E2 * E5) / 68
        )
    )
    return 3 * sum_final + result


_RD0State: TypeAlias = tuple[Array, Array, Array, Array]


def _elliprd_one_zero_iter(y: Array, z: Array) -> Array:
    """Iterative solution for RD when one argument is zero"""
    x0 = jnp.sqrt(y)
    y0 = jnp.sqrt(z)
    sum_term = jnp.zeros_like(y)
    sum_pow = jnp.full_like(y, 0.25)

    def cond_fn(state: _RD0State):
        xn, yn, _, _ = state
        return jnp.abs(xn - yn) >= 2.7 * jnp.finfo(y.dtype).eps * jnp.abs(xn)

    def body_fn(state: _RD0State) -> _RD0State:
        xn, yn, sum_term, sum_pow = state
        t = jnp.sqrt(xn * yn)
        x_new = (xn + yn) / 2
        y_new = t
        sum_pow_new = sum_pow * 2
        tmp = x_new - y_new
        sum_term_new = sum_term + sum_pow_new * tmp * tmp
        return x_new, y_new, sum_term_new, sum_pow_new

    x_final, y_final, sum_final, _ = lax.while_loop(
        cond_fn, body_fn, (x0, y0, sum_term, sum_pow)
    )

    rf = jnp.pi / (x_final + y_final)
    pt = (x0 + 3 * y0) / (4 * z * (x0 + y0))
    pt = pt - sum_final / (z * (y - z))
    return pt * rf * 3


def elliprd(x: Array, y: Array, z: Array) -> Array:
    r"""Carlson symmetric elliptic integral of the second kind

    $$ R_D(x, y, z) = \frac{3}{2} \int_0^\infty \frac{dt}{(t+z) \sqrt{(t+x)(t+y)(t+z)}} $$

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z > 0)

    Returns:
        R_D(x, y, z)
    """
    return lax.select(
        (x == 0.0) | (y == 0.0),
        _elliprd_one_zero_iter(jnp.maximum(x, y), z),
        _elliprd_general_iter(x, y, z),
    )


def elliprc(x: Array, y: Array) -> Array:
    r"""Carlson's degenerate elliptic integral R_C

    $$ R_C(x, y) = \frac{1}{2} \int_0^\infty \frac{dt}{\sqrt{t+x}(t+y)} $$

    This routine does not handle the singular case y < 0

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y > 0)

    Returns:
        R_C(x, y)
    """
    arg = jnp.sqrt((x - y) / x)
    return lax.select(
        x == 0.0,
        jnp.pi / (2 * jnp.sqrt(y)),
        lax.select(
            x == y,
            1.0 / jnp.sqrt(x),
            lax.select(
                y > x,
                jnp.atan(jnp.sqrt((y - x) / x)) / jnp.sqrt(y - x),
                lax.select(
                    y / x > 0.5,
                    (jnp.log1p(arg) - jnp.log1p(-arg)) / (2 * jnp.sqrt(x - y)),
                    jnp.log((jnp.sqrt(x) + jnp.sqrt(x - y)) / jnp.sqrt(y))
                    / jnp.sqrt(x - y),
                ),
            ),
        ),
    )


def elliprc1p(x: Array) -> Array:
    """R_C(1, 1+x) special case

    Does not handle x <= -1 singular case

    Args:
        x: Real argument (x > -1)
    """
    arg = jnp.sqrt(-x)
    return lax.select(
        x == 0.0,
        jnp.ones_like(x),
        lax.select(
            x > 0.0,
            jnp.atan(jnp.sqrt(x)) / jnp.sqrt(x),
            lax.select(
                x > -0.5,
                (jnp.log1p(arg) - jnp.log1p(-arg)) / (2 * arg),
                jnp.log((1 + arg) / jnp.sqrt(1 + x)) / arg,
            ),
        ),
    )


_RJState: TypeAlias = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array
]


def elliprj(x: Array, y: Array, z: Array, p: Array) -> Array:
    r"""Carlson symmetric elliptic integral of the third kind

    $$ R_J(x, y, z, p) = \frac{3}{2} \int_0^\infty \frac{dt}{(t+p) \sqrt{(t+x)(t+y)(t+z)}} $$

    Note: $R_J(x, y, z, z) = R_D(x, y, z)$

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z >= 0)
        p: Real argument (p > 0)

    Returns:
        R_J(x, y, z, p)
    """

    def cond_fun(state: _RJState):
        _, _, _, _, An, _, Q, fmn, _ = state
        return fmn * Q >= An

    def body_fun(state: _RJState) -> _RJState:
        xn, yn, zn, pn, An, delta, Q, fmn, sum_term = state
        rx = jnp.sqrt(xn)
        ry = jnp.sqrt(yn)
        rz = jnp.sqrt(zn)
        rp = jnp.sqrt(pn)
        Dn = (rp + rx) * (rp + ry) * (rp + rz)
        En = (delta / Dn) / Dn
        sum_term_new = lax.select(
            (En > -1.5) & (En < -0.5),
            sum_term
            + fmn
            / Dn
            * elliprc(jnp.ones_like(x), 2 * rp * (pn + rx * (ry + rz) + ry * rz) / Dn),
            sum_term + fmn / Dn * elliprc1p(En),
        )
        lam = rx * ry + rx * rz + ry * rz

        An_new = (An + lam) / 4
        fmn_new = fmn / 4

        # break happens here in boost, but an extra iteration of these vars is OK
        xn_new = (xn + lam) / 4
        yn_new = (yn + lam) / 4
        zn_new = (zn + lam) / 4
        pn_new = (pn + lam) / 4
        delta_new = delta / 64
        return (
            xn_new,
            yn_new,
            zn_new,
            pn_new,
            An_new,
            delta_new,
            Q,
            fmn_new,
            sum_term_new,
        )

    An0 = (x + y + z + 2 * p) / 5
    delta0 = (p - x) * (p - y) * (p - z)
    Q = jnp.pow(jnp.finfo(x).eps / 5, -1 / 8) * jnp.max(
        jnp.array(
            [jnp.abs(An0 - x), jnp.abs(An0 - y), jnp.abs(An0 - z), jnp.abs(An0 - p)]
        )
    )
    fmn0 = jnp.ones_like(x)
    sum_term0 = jnp.zeros_like(x)
    (
        _,
        _,
        _,
        _,
        An_final,
        _,
        _,
        fmn_final,
        sum_final,
    ) = lax.while_loop(
        cond_fun, body_fun, (x, y, z, p, An0, delta0, Q, fmn0, sum_term0)
    )

    X = fmn_final * (An0 - x) / An_final
    Y = fmn_final * (An0 - y) / An_final
    Z = fmn_final * (An0 - z) / An_final
    P = -0.5 * (X + Y + Z)
    E2 = X * Y + X * Z + Y * Z - 3 * P * P
    E3 = X * Y * Z + 2 * E2 * P + 4 * P * P * P
    E4 = (2 * X * Y * Z + E2 * P + 3 * P * P * P) * P
    E5 = X * Y * Z * P * P
    result = (
        fmn_final
        * jnp.pow(An_final, -3 / 2)
        * (
            1
            - 3 * E2 / 14
            + E3 / 6
            + 9 * E2 * E2 / 88
            - 3 * E4 / 22
            - 9 * E2 * E3 / 52
            + 3 * E5 / 26
            - E2 * E2 * E2 / 16
            + 3 * E3 * E3 / 40
            + 3 * E2 * E4 / 20
            + 45 * E2 * E2 * E3 / 272
            - 9 * (E3 * E4 + E2 * E5) / 68
        )
    )
    return result + 6 * sum_final


def elliptic_kepi(n: Array, k: Array) -> tuple[Array, Array, Array]:
    """Compute elliptic integrals of the first, second, and third kind (K, E, Pi)

    Doing it once using Carlson forms may save a little bit of time
    https://en.wikipedia.org/wiki/Carlson_symmetric_form#Complete_elliptic_integrals

    Args:
        n: Parameter for the third kind integral
        k: Modulus (0 <= k < 1)

    Returns:
        K, E, Pi: Complete elliptic integrals of the first, second, and third kind
    """
    zero = jnp.zeros_like(k)
    one = jnp.ones_like(k)
    # Rf = elliprf(zero, 1 - k**2, one)
    Rf = _elliprf_one_zero_iter(1 - k**2, one)
    # Rd = elliprd(zero, 1 - k**2, one)
    Rd = _elliprd_one_zero_iter(1 - k**2, one)
    Rj = elliprj(zero, 1 - k**2, one, 1 - n)
    K = Rf
    E = Rf - k**2 / 3 * Rd
    Pi = Rf + n / 3 * Rj
    return K, E, Pi
