"""Elliptic integrals

Needed mainly for solenoids
"""
from scipy.special import elliprd, elliprf, elliprj


def elliptic_kepi(n, k):
    """Compute elliptic integrals of the first, second, and third kind (K, E, Pi)

    Doing it once using Carlson forms may save a little bit of time
    https://en.wikipedia.org/wiki/Carlson_symmetric_form#Complete_elliptic_integrals
    """
    Rf = elliprf(0, 1 - k**2, 1)
    Rd = elliprd(0, 1 - k**2, 1)
    Rj = elliprj(0, 1 - k**2, 1, 1 - n)
    K = Rf
    E = Rf - k**2 / 3 * Rd
    Pi = Rf + n / 3 * Rj
    return K, E, Pi