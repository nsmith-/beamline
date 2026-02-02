# Beamline

A beamline simulator for particle physics applications with support for both NumPy and JAX backends.

## Project Structure

```
beamline/
├── src/beamline/          # Main package source code
│   ├── units.py           # Physical units handling with Pint and HEPUnits
│   ├── numpy/             # NumPy-based implementations
│   │   ├── bessel.py      # Bessel function implementations
│   │   ├── convolve.py    # Convolution operations
│   │   ├── elliptic.py    # Elliptic integrals
│   │   ├── emfield.py     # Electromagnetic field definitions
│   │   ├── integrate.py   # Integration utilities
│   │   ├── kinematics.py  # Particle kinematics and state
│   │   ├── material.py    # Material properties
│   │   ├── multipole.py   # Multipole field expansions
│   │   ├── pillbox.py     # Pillbox cavity simulations
│   │   └── solenoid.py    # Solenoid magnet models
│   └── jax/               # JAX-based implementations (optional, differentiable)
│       ├── bessel.py      # JAX Bessel functions
│       ├── elliptic.py    # JAX elliptic integrals
│       ├── integrators.py # JAX integration methods
│       ├── stencils.py    # Finite difference stencils
│       ├── types.py       # JAX type definitions
│       └── magnet/        # Magnet models
│           ├── loop.py    # Current loop models
│           └── solenoid.py # JAX solenoid implementations
└── test/                  # Test suite
    ├── numpy/             # Tests for NumPy implementations
    ├── jax/               # Tests for JAX implementations
    └── test_units.py      # Tests for unit handling

```

### Core Components

- **units.py**: Defines unit handling using Pint with HEP-specific units from HEPUnits, including conversions to/from CLHEP units
- **numpy/**: Standard NumPy implementations for electromagnetic field calculations, particle kinematics, and beamline element simulations
- **jax/** (optional): JAX-based implementations that support automatic differentiation and JIT compilation for optimization and sensitivity studies

### Backend Implementations

The project provides two backend implementations:

1. **NumPy Backend** (`beamline.numpy`): Standard implementation using NumPy and SciPy for numerical computations
2. **JAX Backend** (`beamline.jax`): Optional implementation using JAX for automatic differentiation and GPU acceleration (requires `beamline[jax]` installation)

Both backends provide similar functionality but with different performance characteristics and capabilities.
