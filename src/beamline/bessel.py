"""Some extra Bessel function helpers"""

import numpy as np
from scipy.special import iv, kv, jv, gamma


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


def jv_over_z(v, z):
    """Compute j_v(z) / z with care around z=0

    Uses asymptotic expansion for small z
    """
    cutoff = 1e-7
    return np.where(
        z < cutoff,
        np.pow(z, v - 1) / (2**v * gamma(v + 1)),
        jv(v, z) / np.maximum(z, cutoff),
    )


def _jv_over_z_study(v: int = 1, cutoff: float = 1e-7):
    """Study of jv(z)/z implementation, to settle on cutoff value

    At 1e-7, we start to see the switch to asymptotic form in the scipy jv implementation
    """
    import matplotlib.pyplot as plt

    z = np.linspace(1e-4 * cutoff, 3 * cutoff, 100)

    jvz = jv(v, z) / z
    laurent = np.pow(z, v - 1) / (2**v * gamma(v + 1))
    res = np.where(z < cutoff, laurent, jvz)

    # using recurrence relation (not as good)
    alt = (jv(v - 1, z) - jv(v + 1, z)) / (2 * v)

    fig, ax = plt.subplots()
    ax.plot(z, (res - jvz) / jvz, label="via asymptotic")
    ax.plot(z, (alt - jvz) / jvz, "--", label="via recurrence")
    ax.legend()
    return ax
