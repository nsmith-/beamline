import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import gamma, jv


@pytest.mark.parametrize("v", [1, 2])
@pytest.mark.parametrize("cutoff", [1e-6, 1e-7, 1e-8])
def test_jv_over_z(artifacts_dir, v: int, cutoff: float):
    """Study of jv(z)/z implementation, to settle on cutoff value

    At 1e-7, we start to see the switch to asymptotic form in the scipy jv implementation
    """

    z = np.geomspace(1e-4 * cutoff, 1e1 * cutoff, 100)

    jvz = jv(v, z) / z
    laurent = np.pow(z, v - 1) / (2**v * gamma(v + 1))
    res = np.where(z < cutoff, laurent, jvz)

    # using recurrence relation (not as good)
    alt = (jv(v - 1, z) - jv(v + 1, z)) / (2 * v)

    fig, ax = plt.subplots()
    ax.plot(z, (res - jvz) / jvz, label="via asymptotic")
    ax.plot(z, (alt - jvz) / jvz, "--", label="via recurrence")
    ax.legend()
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / f"jv_over_z_v{v}_cutoff{cutoff:.0e}.png")

