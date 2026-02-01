"""Elliptic integrals

This module implements Carlson's symmetric elliptic integrals R_F, R_D, R_J, R_C, and R_G
using JAX, with custom jvp rules for automatic differentiation.

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

import jax
import jax.numpy as jnp
from jax import lax

from beamline.jax.types import SFloat

_RFState: TypeAlias = tuple[SFloat, SFloat, SFloat, SFloat, SFloat, SFloat]
"""x, y, z, A, Q, f"""


@jax.custom_jvp
def _elliprf_full_iter(x: SFloat, y: SFloat, z: SFloat) -> SFloat:
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


@_elliprf_full_iter.defjvp
def _elliprf_full_iter_fwd(primals, tangents):
    x, y, z = primals
    dx, dy, dz = tangents
    rf = _elliprf_full_iter(x, y, z)

    drf_dx = -elliprd(y, z, x) / 6
    drf_dy = -elliprd(x, z, y) / 6
    drf_dz = -elliprd(x, y, z) / 6

    drf = drf_dx * dx + drf_dy * dy + drf_dz * dz
    return rf, drf


_RF0State: TypeAlias = tuple[SFloat, SFloat]
"""x, y"""


@jax.custom_jvp
def _elliprf_one_zero_iter(x: SFloat, y: SFloat) -> SFloat:
    """Solution for the case where one argument is zero"""
    xn = jnp.sqrt(x)
    yn = jnp.sqrt(y)

    def cond_fn(state: _RF0State):
        x, y = state
        return jnp.abs(x - y) >= 2.7 * jnp.sqrt(jnp.finfo(x.dtype).eps) * jnp.abs(x)

    def body_fn(state: _RF0State) -> _RF0State:
        x, y = state
        t = jnp.sqrt(x * y)
        x_new = (x + y) / 2
        y_new = t
        return x_new, y_new

    x_final, y_final = lax.while_loop(cond_fn, body_fn, (xn, yn))
    return jnp.pi / (x_final + y_final)


@_elliprf_one_zero_iter.defjvp
def _elliprf_one_zero_iter_fwd(primals, tangents):
    x, y = primals
    dx, dy = tangents
    rf = _elliprf_one_zero_iter(x, y)

    drf_dx = -elliprd_one_zero(y, x) / 6
    drf_dy = -elliprd_one_zero(x, y) / 6

    drf = drf_dx * dx + drf_dy * dy
    return rf, drf


def elliprf_one_zero(x: SFloat, y: SFloat) -> SFloat:
    """Special case of Carlson R_F where one argument is zero

    Args:
        x: Real argument (x > 0)
        y: Real argument (y > 0)

    Returns:
        R_F(x, y, 0)

    Implementation notes:
    Originally we selected pi/2/sqrt(x) when x == y, but this seems
    to cause trouble for autograd.
    """
    return _elliprf_one_zero_iter(x, y)


def elliprf(x: SFloat, y: SFloat, z: SFloat) -> SFloat:
    r"""Carlson symmetric elliptic integral of the first kind

    $$ R_F(x, y, z) = \frac{1}{2} \int_0^\infty \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}} $$

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z >= 0)

    At most one of x, y, z can be zero.

    Returns:
        R_F(x, y, z)

    Implementation notes:
    Originally we selected elliprc when x == y or y == z,
    but this seems to cause trouble for autograd, and the iterative
    method gives the same result in those cases
    """
    lo, me, hi = jnp.sort(jnp.array([x, y, z]))
    losafe = lax.select(lo == 0.0, 1e-16, lo)
    return lax.select(
        lo == 0.0,
        elliprf_one_zero(me, hi),
        _elliprf_full_iter(losafe, me, hi),
    )


_RDState: TypeAlias = tuple[SFloat, SFloat, SFloat, SFloat, SFloat, SFloat, SFloat]
"""x, y, z, A, Q, sum_term, f"""


@jax.custom_jvp
def _elliprd_general_iter(x: SFloat, y: SFloat, z: SFloat) -> SFloat:
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


def _drd_dx_term1(x, y):
    xmy = lax.select(x == y, 1e-16, x - y)
    return (
        3 / (16 * xmy**2) * (jnp.sqrt(y) * (5 * x - 2 * y) / x**2 - 3 * elliprc(y, x))
    )


def _drd_dx_term2(x, y, z, rd):
    xmz = lax.select(x == z, 1e-16, x - z)
    return (_elliprd_general_iter(y, z, x) - rd) / (2 * xmz)


def _drd_dx_switch(x, y, z, rd):
    return lax.select(
        (x == y) & (y == z),
        -3 / 10 / x**2 / jnp.sqrt(x),
        lax.select(x == z, _drd_dx_term1(x, y), _drd_dx_term2(x, y, z, rd)),
    )


def _drd_dz_xeqz(x, y, z, rd):
    ymz = lax.select(y == z, 1e-16, y - z)
    return -9 / 16 / ymz**2 * (3 * elliprc(y, z) + jnp.sqrt(y) * (2 * y - 5 * z) / z**2)


def _drd_dz_else(x, y, z, rd):
    xmz = lax.select(x == z, 1e-16, x - z)
    ymz = lax.select(y == z, 1e-16, y - z)
    return (
        (x + y - 2 * z) * rd
        + 3 / 2 * elliprf(x, y, z)
        - 3 / 2 * jnp.sqrt(x * y) / z / jnp.sqrt(z)
    ) / (xmz * ymz)


@_elliprd_general_iter.defjvp
def _elliprd_general_iter_fwd(primals, tangents):
    x, y, z = primals
    dx, dy, dz = tangents
    rd_xyz = _elliprd_general_iter(x, y, z)

    drd_dx = _drd_dx_switch(x, y, z, rd_xyz)
    drd_dy = _drd_dx_switch(y, x, z, rd_xyz)
    drd_dz = lax.select(
        (x == y) & (y == z),
        -9 / 10 / z**2 / jnp.sqrt(z),
        lax.select(
            x == z,
            _drd_dz_xeqz(x, y, z, rd_xyz),
            lax.select(
                y == z,
                _drd_dz_xeqz(y, x, z, rd_xyz),
                _drd_dz_else(x, y, z, rd_xyz),
            ),
        ),
    )

    drd = drd_dx * dx + drd_dy * dy + drd_dz * dz
    return rd_xyz, drd


_RD0State: TypeAlias = tuple[SFloat, SFloat, SFloat, SFloat]
"""x, y, sum_term, sum_pow"""


def _elliprd_one_zero_iter(y: SFloat, z: SFloat) -> SFloat:
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


def elliprd_one_zero(y: SFloat, z: SFloat) -> SFloat:
    """Special case of Carlson R_D where one argument is zero

    Args:
        y: Real argument (y > 0)
        z: Real argument (z > 0)

    Returns:
        R_D(0, y, z)
    """
    return _elliprd_one_zero_impl(y, z)


@jax.custom_jvp
def _elliprd_one_zero_impl(y: SFloat, z: SFloat) -> SFloat:
    # guard against nan for y == z in the second branch
    ysafe = jnp.where(y == z, y + z, y)
    return lax.select(
        y == z,
        3 * jnp.pi / (4 * y * jnp.sqrt(y)),
        _elliprd_one_zero_iter(ysafe, z),
    )


@_elliprd_one_zero_impl.defjvp
def _elliprd_one_zero_impl_fwd(primals, tangents):
    y, z = primals
    ysafe = jnp.where(y == z, y + z, y)
    dy, dz = tangents
    rd_yz = _elliprd_one_zero_impl(y, z)
    rd_zy = _elliprd_one_zero_impl(z, y)

    drd_dy = lax.select(
        y == z,
        -9 * jnp.pi / 32 / y**2 / jnp.sqrt(y),
        (rd_zy - rd_yz) / 2 / (ysafe - z),
    )
    drd_dz = lax.select(
        y == z,
        -27 * jnp.pi / 32 / z**2 / jnp.sqrt(z),
        ((y - 2 * z) * rd_yz + 3 / 2 * elliprf_one_zero(y, z)) / (z - ysafe) / z,
    )

    drd = drd_dy * dy + drd_dz * dz
    return rd_yz, drd


def elliprd(x: SFloat, y: SFloat, z: SFloat) -> SFloat:
    r"""Carlson symmetric elliptic integral of the second kind

    $$ R_D(x, y, z) = \frac{3}{2} \int_0^\infty \frac{dt}{(t+z) \sqrt{(t+x)(t+y)(t+z)}} $$

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z > 0)

    Returns:
        R_D(x, y, z)
    """
    lo, hi = jnp.minimum(x, y), jnp.maximum(x, y)
    return lax.select(
        lo == 0.0,
        elliprd_one_zero(hi, z),
        _elliprd_general_iter(x, y, z),
    )


def elliprc(x: SFloat, y: SFloat) -> SFloat:
    r"""Carlson's degenerate elliptic integral R_C

    $$ R_C(x, y) = \frac{1}{2} \int_0^\infty \frac{dt}{\sqrt{t+x}(t+y)} $$

    Note: $R_C(x, y) = R_F(x, y, y)$

    This routine does not handle the singular case y < 0

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y > 0)

    Returns:
        R_C(x, y)
    """
    # even when we don't select a branch, we'd like it to not produce NaNs
    xsafe = lax.select(x == 0.0, 1e-16, x)
    sqrt_abs = jnp.sqrt(abs(x - y))
    sqrt_relx = jnp.sqrt(abs(x - y) / xsafe)
    # we only go in the log1p branch when 0.5 < y/x <= 1
    sqrt_relx_safe = lax.select(sqrt_relx >= 1.0, jnp.zeros_like(x), sqrt_relx)
    return lax.select(
        x == 0.0,
        jnp.pi / (2 * jnp.sqrt(y)),
        lax.select(
            x == y,
            1.0 / jnp.sqrt(xsafe),
            lax.select(
                y > x,
                jnp.atan(sqrt_relx) / sqrt_abs,
                lax.select(
                    y / xsafe > 0.5,
                    (jnp.log1p(sqrt_relx) - jnp.log1p(-sqrt_relx_safe))
                    / (2 * sqrt_abs),
                    jnp.log((jnp.sqrt(x) + sqrt_abs) / jnp.sqrt(y)) / sqrt_abs,
                ),
            ),
        ),
    )


def elliprc1p(x: SFloat) -> SFloat:
    """R_C(1, 1+x) special case

    Does not handle x <= -1 singular case

    Args:
        x: Real argument (x > -1)
    """
    xsafe = lax.select(x == 0.0, 1e-16, x)
    arg = jnp.sqrt(abs(xsafe))
    return lax.select(
        x == 0.0,
        jnp.ones_like(x),
        lax.select(
            x > 0.0,
            jnp.atan(arg) / arg,
            lax.select(
                x > -0.5,
                (jnp.log1p(arg) - jnp.log1p(jnp.maximum(-1.0, -arg))) / (2 * arg),
                jnp.log((1 + arg) / jnp.sqrt(1 + x)) / arg,
            ),
        ),
    )


def elliprj(x: SFloat, y: SFloat, z: SFloat, p: SFloat) -> SFloat:
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
    return _elliprj_impl(x, y, z, p)


_RJState: TypeAlias = tuple[
    SFloat, SFloat, SFloat, SFloat, SFloat, SFloat, SFloat, SFloat, SFloat
]
"""x, y, z, p, A, delta, Q, fmn, sum_term

Q is unaltered so could be captured instead"""


@jax.custom_jvp
def _elliprj_impl(x: SFloat, y: SFloat, z: SFloat, p: SFloat) -> SFloat:
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


def elliprg(x: SFloat, y: SFloat, z: SFloat) -> SFloat:
    r"""Carlson symmetric elliptic integral R_G

    $$ R_G(x, y, z) = \frac{1}{4} \int_0^\infty \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}} \left( \frac{x}{t+x} + \frac{y}{t+y} + \frac{z}{t+z} \right) $$

    This can be computed from R_F and R_D as:
    $$ R_G(x, y, z) = \frac{1}{2} \left( z R_F(x, y, z) - \frac{1}{3} (x-z)(y-z) R_D(x, y, z) + \sqrt{\frac{xy}{z}} \right) $$
    where we permute z to be nonzero

    ([1], Eq. 7)

    Args:
        x: Real argument (x >= 0)
        y: Real argument (y >= 0)
        z: Real argument (z >= 0)

    Note: assumes at most one of x, y, z is zero. R_G is defined for more zeros but
    this implementation does not cover those cases.

    Returns:
        R_G(x, y, z)

    [1] https://arxiv.org/pdf/math/9409227
    """
    # ensure z > 0 for R_D
    y, z = jnp.minimum(y, z), jnp.maximum(y, z)
    return 0.5 * (
        z * elliprf(x, y, z)
        - (x - z) * (y - z) / 3 * elliprd(x, y, z)
        + jnp.sqrt(x * y / z)
    )


def _drj_dx_term_xyp(x, y, z):
    num = 3 * jnp.sqrt(z) * (5 * x - 2 * z) - 9 * x**2 * elliprc(z, x)
    den = 16 * (x - z) ** 2 * x**2
    return num / lax.select(x == z, 1.0, den)


def _drj_dx_term_xp(x, y, z):
    num = (
        2 / 3 * (y + z - 2 * x) * elliprd(y, z, x)
        + elliprf(x, y, z)
        - jnp.sqrt(y * z / x**3)
    )
    den = 2 * (x - y) * (x - z)
    return num / lax.select((x == y) | (x == z), 1.0, den)


def _drj_dx_term_else(x, y, z, p, rj):
    num = rj - elliprd(y, z, x)
    den = 2 * (p - x)
    return num / lax.select(x == p, 1.0, den)


def _drj_dx_switch(x, y, z, p, rj):
    # for x == 0.0, we evaluate R_C and R_D out of bounds in some branches
    x0 = lax.select(x == 0.0, 1e-16, x)
    return lax.select(
        (x == y) & (y == z) & (z == p),
        -3 / 10 / x**2 / jnp.sqrt(x),
        lax.select(
            (x == y) & (y == p),
            _drj_dx_term_xyp(x0, y, z),
            lax.select(
                (x == z) & (z == p),
                _drj_dx_term_xyp(x0, z, y),
                lax.select(
                    x == p,
                    _drj_dx_term_xp(x0, y, z),
                    _drj_dx_term_else(x0, y, z, p, rj),
                ),
            ),
        ),
    )


def _drj_dp_term_2eq(p, x):
    "p=y=z case"
    num = -3 * (jnp.sqrt(x) * (-5 * p + 2 * x) / p**2 + 3 * elliprc(x, p))
    den = 8 * (p - x) ** 2
    return num / lax.select(p == x, 1.0, den)


def _drj_dp_term_1eq(p, y, z):
    "p=x (or p=y or p=z) case"
    num = (
        -(jnp.sqrt(y * z) / p ** (3 / 2))
        + (2 * (-2 * p + y + z) * elliprd(y, z, p)) / 3
        + elliprf(p, y, z)
    )
    den = (p - y) * (p - z)
    return num / lax.select((p == y) | (p == z), 1.0, den)


def _drj_dp_term_else(x, y, z, p, rj):
    term1_num = 3 * (
        jnp.sqrt(x * y * z) / p + p * elliprf(x, y, z) - 2 * elliprg(x, y, z)
    )
    pmx, pmy, pmz = p - x, p - y, p - z
    term2_num = -(pmy * pmz + pmz * pmx + pmx * pmy) * rj
    den = 2 * pmx * pmy * pmz
    return (term1_num + term2_num) / lax.select(den == 0.0, 1.0, den)


def _twiddling(x, y, z, p):
    """Some bit twiddling to return values of x,y,z that are not equal to p"""
    peqx, peqy, peqz = (
        jnp.astype(p == x, jnp.int8),
        jnp.astype(p == y, jnp.int8),
        jnp.astype(p == z, jnp.int8),
    )
    neq = peqx + peqy + peqz
    # for n=2 this is the odd one out
    arg = (neq == 2) * ((peqy & peqz) * x + (peqx & peqz) * y + (peqx & peqy) * z)
    arg = arg + (neq == 1) * (peqx * y + peqy * z + peqz * x)
    # for n=1 grab the other unique one
    arg2 = (neq == 1) * (peqx * z + peqy * x + peqz * y)
    # avoid returning zeros
    arg2 = arg2 + (neq != 1) * 1.0
    return neq, arg, arg2


def _drj_dp_switch(x, y, z, p, rj):
    """
    Piecewise[{
            {-3/(5 p^(5/2)), p == x == y == z},
            {(-3 ((Sqrt[x] (-5 p + 2 x))/p^2 + 3 CarlsonRC[x, p]))/(8 (p - x)^2), (p == x == y && p != z) || (p != x && p == y && y == z)},
            {(-3 ((Sqrt[y] (-5 p + 2 y))/p^2 + 3 CarlsonRC[y, p]))/(8 (p - y)^2), p == x == z && p != y},
            {(-((Sqrt[y] Sqrt[z])/p^(3/2)) + (2 (-2 p + y + z) CarlsonRD[y, z, p])/3 + CarlsonRF[p, y, z])/((p - y) (p - z)), p == x && p != y && p != z},
            {(-((Sqrt[x] Sqrt[z])/p^(3/2)) + (2 (-2 p + x + z) CarlsonRD[x, z, p])/3 + CarlsonRF[p, x, z])/((p - x) (p - z)), p == y && p != x && p != z},
            {(-((Sqrt[x] Sqrt[y])/p^(3/2)) + (2 (-2 p + x + y) CarlsonRD[x, y, p])/3 + CarlsonRF[p, x, y])/((p - x) (p - y)), p == z && p != x && p != y}
        },
        ((3 ((Sqrt[x] Sqrt[y] Sqrt[z])/p + p CarlsonRF[x, y, z] - 2 CarlsonRG[x, y, z]))/((p - x) (p - y) (p - z)) - ((p - x)^(-1) + (p - y)^(-1) + (p - z)^(-1)) CarlsonRJ[x, y, z, p])/2
    ]
    """
    neq, arg, arg2 = _twiddling(x, y, z, p)
    return lax.select(
        neq == 3,
        -3 / (5 * p**2 * jnp.sqrt(p)),
        lax.select(
            neq == 2,
            _drj_dp_term_2eq(p, arg),
            lax.select(
                neq == 1,
                _drj_dp_term_1eq(p, arg, arg2),
                _drj_dp_term_else(x, y, z, p, rj),
            ),
        ),
    )


@_elliprj_impl.defjvp
def _elliprj_impl_fwd(primals, tangents):
    x, y, z, p = primals
    dx, dy, dz, dp = tangents
    rj = _elliprj_impl(x, y, z, p)

    drj_dx = _drj_dx_switch(x, y, z, p, rj)
    drj_dy = _drj_dx_switch(y, x, z, p, rj)
    drj_dz = _drj_dx_switch(z, x, y, p, rj)
    drj_dp = _drj_dp_switch(x, y, z, p, rj)

    drj = drj_dx * dx + drj_dy * dy + drj_dz * dz + drj_dp * dp
    return rj, drj


@jax.custom_jvp
def _elliptic_kepi_imp(n: SFloat, k: SFloat) -> tuple[SFloat, SFloat, SFloat]:
    # TODO: protect against k = 1 (unterminated loop)
    n, k = jnp.broadcast_arrays(n, k)
    zero = jnp.zeros_like(k)
    one = jnp.ones_like(k)
    # TODO: k = 0 is found in wire loop and solenoid on axis, need better handling
    # Rf = elliprf(zero, 1 - k**2, one)
    Rf = elliprf_one_zero(1 - k**2, one)
    # Rd = elliprd(zero, 1 - k**2, one)
    Rd = elliprd_one_zero(1 - k**2, one)
    Rj = elliprj(zero, 1 - k**2, one, 1 - n)
    K = Rf
    E = Rf - k**2 / 3 * Rd
    Pi = Rf + n / 3 * Rj
    return K, E, Pi


def elliptic_kepi(n: SFloat, k: SFloat) -> tuple[SFloat, SFloat, SFloat]:
    """Compute elliptic integrals of the first, second, and third kind (K, E, Pi)

    Doing it once using Carlson forms may save a little bit of time
    https://en.wikipedia.org/wiki/Carlson_symmetric_form#Complete_elliptic_integrals

    Args:
        n: Parameter for the third kind integral (n < 1, typically n > -1)
        k: Modulus (0 <= k < 1)

    Returns:
        K, E, Pi: Complete elliptic integrals of the first, second, and third kind
    """
    return _elliptic_kepi_imp(n, k)


@_elliptic_kepi_imp.defjvp
def elliptic_kepi_fwd(primals, tangents):
    n, k = primals
    # "double-where trick" ensure autograd handles k = 0 and n = 0 correctly
    # by sanitizing the inputs for the non-limiting case branches
    n0, k0 = lax.select(n == 0.0, 0.1234, n), lax.select(k == 0.0, 0.1234, k)
    dn, dk = tangents
    K, E, Pi = _elliptic_kepi_imp(n, k)
    # https://en.wikipedia.org/wiki/Elliptic_integral#Differential_equation
    dK_dk = lax.select(
        k == 0.0,
        0.0,
        # (k0 * K - K + E) / (2 * k0 * (k0 - 1)),
        E / (k0 * (1 - k0**2)) - K / k0,
    )
    # https://en.wikipedia.org/wiki/Elliptic_integral#Derivative_and_differential_equation
    dE_dk = lax.select(
        k == 0.0,
        0.0,
        (E - K) / k0,
    )
    # https://en.wikipedia.org/wiki/Elliptic_integral#Partial_derivatives
    dPi_dk = lax.select(
        k == 0.0,
        0.0,
        k0 / (n - k0**2) * (E / (k0**2 - 1) + Pi),
    )
    dPi_dn = lax.select(
        n == 0.0,
        lax.select(
            k == 0.0,
            0.0,
            # 1 / -2k**2 (E - K + k**2/n * (K - Pi))
            # (K - Pi) = -n/3 * Rj
            # => E - K - k**2/3 * Rj = 2*(E - K)
            # as Rd = Rj at n = 0
            (K - E) / k0**2,
        ),
        (
            1
            / (2 * (k**2 - n0) * (n0 - 1))
            * (E + (k**2 - n0) * K / n0 + (n0**2 - k**2) * Pi / n0)
        ),
    )
    return (K, E, Pi), (
        dK_dk * dk,
        dE_dk * dk,
        dPi_dk * dk + dPi_dn * dn,
    )
