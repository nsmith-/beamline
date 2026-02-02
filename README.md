# Beamline

A beamline simulator for particle physics applications with support for both NumPy and JAX backends.

## Features

- **Physical units handling**: Integration with Pint and HEPUnits for proper dimensional analysis, including conversions to/from CLHEP units
- **Bessel function implementations**: Special functions for field calculations
- **Convolution operations**: Signal processing utilities
- **Elliptic integrals**: Mathematical functions for electromagnetic field calculations
- **Electromagnetic field definitions**: Field strength and tensor representations
- **Integration utilities**: Numerical integration methods
- **Particle kinematics and state**: Relativistic particle dynamics
- **Material properties**: Material interaction models
- **Multipole field expansions**: Magnetic field multipole representations
- **Pillbox cavity simulations**: RF cavity models
- **Solenoid magnet models**: Detailed solenoid field calculations
- **Finite difference stencils**: Numerical differentiation methods
- **Current loop models**: Electromagnetic field from current loops

### Backend Implementations

The project provides two backend implementations:

1. **NumPy Backend** (`beamline.numpy`): Standard implementation using NumPy and SciPy for numerical computations
2. **JAX Backend** (`beamline.jax`): Optional implementation using JAX for automatic differentiation and GPU acceleration (requires `beamline[jax]` installation)

Both backends provide similar functionality but with different performance characteristics and capabilities.
