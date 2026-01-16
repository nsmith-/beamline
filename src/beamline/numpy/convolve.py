import logging

import numpy as np

logger = logging.getLogger(__name__)


def fftconv(
    probfun, support: tuple[float, float], n: float, fftlen: int = 2**10, foldn: int = 8
):
    """Convolve a probability distribution function with itself n times using FFT.

    Uses a recursive folding strategy to handle large n without running into numerical issues in
    the exponentiation. As a result, for best results, n // foldn should be an integer. If not,
    the first convolution will be fractional and may introduce some artifacts.

    Args:
        probfun: Function that takes a numpy array and returns the probability density at those points.
        support: Tuple (xmin, xmax) defining the support of the probability distribution.
        n: Number of times to convolve the distribution with itself.
        fftlen: Length of the FFT to use (should be a power of two).
        foldn: Number of convolutions to perform in each folding step (should be a power of two).
    """
    if fftlen & (fftlen - 1) != 0:
        raise ValueError("fftlen must be a power of two")
    if foldn & (foldn - 1) != 0:
        raise ValueError("foldn must be a power of two")
    xmin, xmax = support

    seq = [
        min(foldn, n / (foldn**i))
        for i in range(int(np.ceil(np.log(n) / np.log(foldn))))
    ]
    # we want to do the fractional convolution first (if any)
    seq.reverse()

    nconv = 1
    domain = np.linspace(xmin, xmin + (xmax - xmin) * seq[0], fftlen)
    logger.debug(
        f"Evaluating probfun on domain {domain[0]} - {domain[domain <= xmax][-1]}"
    )
    work = probfun(domain)
    work[domain > xmax] = 0.0
    fftwork = np.zeros(fftlen // 2 + 1, dtype=np.complex128)

    for i, conv_this_time in enumerate(seq):
        nconv *= conv_this_time
        logger.debug(f"Convolving {conv_this_time} times (total {nconv})")
        work /= work.sum()
        np.fft.rfft(work, out=fftwork)
        np.power(fftwork, conv_this_time, out=fftwork)
        np.fft.irfft(fftwork, out=work)
        if i < len(seq) - 1:
            logger.debug("Downsampling for next round")
            work[: fftlen // foldn] = work[::foldn]
            work[fftlen // foldn :] = 0.0

    work /= work.sum() * (xmax - xmin) * n / fftlen
    domain = np.linspace(nconv * xmin, nconv * xmax, fftlen)
    return work, domain
