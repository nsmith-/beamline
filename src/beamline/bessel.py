"""Some extra Bessel function helpers

"""
from scipy.special import iv, kv


def iv_prime(v, x):
    """Derivative of modified Bessel function of the first kind.

    From https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/ShowAll.html
    """
    if v == 0:
        return iv(1, x)
    # all equivalent forms
    return 0.5 * (iv(v - 1, x) + iv(v + 1, x))
    # v / x * iv(v, x) + iv(v + 1, x)
    # iv(v - 1, x) - v / x * iv(v, x)


def kv_prime(v, x):
    """Derivative of modified Bessel function of the second kind.

    From https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/20/ShowAll.html
    """
    if v == 0:
        return -kv(1, x)
    # all equivalent forms
    return -0.5 * (kv(v - 1, x) + kv(v + 1, x))
    # v / x * kv(v, x) - kv(v + 1, x)
    # -kv(v - 1, x) - v / x * kv(v, x)
