import pint
import hepunits

u = pint.UnitRegistry()

c_light = hepunits.c_light
clhep_length = u.millimeter
clhep_time = u.nanosecond
clhep_mass = u.MeV * clhep_length**-2 * clhep_time**2
clhep_current = u.elementary_charge / clhep_time


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


# Check constants
pint_c = to_clhep(1 * u.speed_of_light)
assert pint_c == hepunits.c_light
pint_h = to_clhep(1 * u.planck_constant)
# See https://github.com/scikit-hep/hepunits/issues/262#issuecomment-3123865095
assert (pint_h - hepunits.h_Planck) / hepunits.h_Planck < 1e-15
pint_e = to_clhep(1 * u.elementary_charge)
assert pint_e == 1.0


# Useful
u.define("MeVc = MeV / speed_of_light")
u.define("GeVc = GeV / speed_of_light")
# PDG: 0.1134289259 +- 0.0000000025 u
# The atomic_mass_constant has a similar relative error as this measurement
u.define("muon_mass = 0.1134289259 * atomic_mass_constant")