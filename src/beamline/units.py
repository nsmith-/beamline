import pint
from hepunits.pint import from_clhep, to_clhep

ureg = pint.UnitRegistry()


def check_dimensionality(
    MeV: int = 0, ns: int = 0, mm: int = 0, e: int = 0, *, expected: pint.Unit
) -> bool:
    """Check that the given combination of CLHEP dimensions matches a given Pint unit dimensionality"""
    return (
        ureg.MeV**MeV
        * ureg.nanosecond**ns
        * ureg.millimeter**mm
        * ureg.elementary_charge**e
    ).dimensionality == expected.dimensionality


# Useful shorthand
ureg.define("MeVc = MeV / speed_of_light")
ureg.define("GeVc = GeV / speed_of_light")
# PDG: 0.1134289259 +- 0.0000000025 u
# The atomic_mass_constant has a similar relative error as this measurement
ureg.define("muon_mass = 0.1134289259 * atomic_mass_constant")

__all__ = ["check_dimensionality", "from_clhep", "to_clhep", "ureg"]
