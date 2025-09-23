import pint
import hepunits

u = pint.UnitRegistry()

c_light = hepunits.c_light
clhep_length = u.millimeter
clhep_time = u.nanosecond
clhep_mass = u.MeV * clhep_length**-2 * clhep_time**2
clhep_current = u.elementary_charge / clhep_time


def check_units(MeV: int, ns: int, mm: int, e: int, expected: pint.Unit) -> bool:
    """Check that the given combination of CLHEP units matches the expected Pint unit"""
    return (
        u.MeV**MeV * u.nanosecond**ns * u.millimeter**mm * u.elementary_charge**e
    ).dimensionality == expected.dimensionality


def to_clhep(quantity: pint.Quantity) -> float:
    """Convert a Pint quantity to a CLHEP quantity"""
    dim_mass = quantity.dimensionality["[mass]"]
    dim_length = quantity.dimensionality["[length]"]
    dim_time = quantity.dimensionality["[time]"]
    dim_current = quantity.dimensionality["[current]"]

    return quantity.to(
        clhep_mass**dim_mass
        * clhep_length**dim_length
        * clhep_time**dim_time
        * clhep_current**dim_current
    ).magnitude


def from_clhep(quantity: float, desired_unit: pint.Unit) -> pint.Quantity:
    """Convert a CLHEP quantity to a Pint quantity"""
    dim_mass = desired_unit.dimensionality["[mass]"]
    dim_length = desired_unit.dimensionality["[length]"]
    dim_time = desired_unit.dimensionality["[time]"]
    dim_current = desired_unit.dimensionality["[current]"]

    pint_quantity = quantity * (
        clhep_mass**dim_mass
        * clhep_length**dim_length
        * clhep_time**dim_time
        * clhep_current**dim_current
    )
    return pint_quantity.to(desired_unit)


# Useful
u.define("MeVc = MeV / speed_of_light")
u.define("GeVc = GeV / speed_of_light")
# PDG: 0.1134289259 +- 0.0000000025 u
# The atomic_mass_constant has a similar relative error as this measurement
u.define("muon_mass = 0.1134289259 * atomic_mass_constant")
