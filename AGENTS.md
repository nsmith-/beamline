# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

`beamline` is an R&D package for **differentiable muon beamline simulation in JAX**, aimed at muon cooling channel design optimization for a muon collider. The physics scope is the intersection of beam transport optics, material interaction (energy loss, multiple scattering, straggling), and collective beam dynamics. The whole design leans on JAX autodiff and GPU acceleration so that beamline parameters (magnet currents, geometry, etc.) can be optimized via gradients through the full simulation.

The project is pre-1.0 ("Planning" status). There is no library API or CLI yet — **results are curated through the test suite**, which writes figures and data into `test_artifacts/`.

## Commands

```bash
uv sync                       # set up venv + install deps (includes jax via dev group)
uv run pytest                 # run all tests; populates ./test_artifacts (takes a few minutes)
uv run pytest test/jax/test_emfield.py            # one file
uv run pytest test/jax/test_emfield.py::test_name # one test
uv run pytest -m extended     # run the slow/extended tests (excluded by default)
uv run ruff check             # lint
uv run ruff format            # format
pre-commit run --all-files    # full lint/format/validation suite
```

Notes:
- `addopts` excludes `extended`-marked tests and auto-saves benchmarks (`pytest-benchmark` → `.benchmarks/`).
- JAX tests are skipped entirely if `jax` is not importable (see `test/jax/conftest.py`), which also enables `jax_debug_nans` for all jax tests — expect failures on any NaN.
- Units throughout are the **CLHEP system** (MeV, ns, mm, e) via `hepunits`; import as `import hepunits as u` and write literals like `250.0 * u.mm`. `src/beamline/units.py` uses `pint` only to *define* derived constants (muon mass/charge, μ₀), then converts to CLHEP floats.

## Two parallel implementations

- `src/beamline/numpy/` — reference/prototype physics in NumPy + SciPy. Use to cross-check the jax versions.
- `src/beamline/jax/` — the real, differentiable implementation. **New work goes here.** Tests mirror this split under `test/numpy/` and `test/jax/`.

`src/beamline/jax/__init__.py` enables float64 (`jax_enable_x64`) on import — always import the package (not bare `jax`) so this runs.

## jax architecture (the part that needs reading multiple files)

Everything is built on **`equinox.Module`** (immutable JAX PyTrees). The core abstractions compose as follows:

- **Coordinates** (`jax/coordinates.py`): `CoordinateChart` subclasses `Cartesian3/4` and `Cylindric3/4` wrap a single `coords` array of shape `(..., N)` (leading dims broadcast). Conversions via `.to_cartesian()` / `.to_cylindric()`. A **`Tangent[T]`** is a (point, vector) pair whose basis vectors are **normalized** so components carry uniform physical units; changing a tangent's chart uses `jax.jvp` and rescaling by Lamé coefficients (`_change_basis`). `GradientField` / `DivergenceField` apply `jax.grad`/`jax.jacobian` with the metric corrections. `Transform` is a 4D rigid transform (rotation + translation) used to place locally-defined fields into global coordinates; `TransformEMField`/`TransformOneForm` wrap a field with a `Transform`.

- **Geometry** (`jax/geometry.py`): `Volume` is the spatial-extent interface — `contains(point)` and `signed_distance(ray)` (signed time-to-surface, `inf` if no hit; min over constituents for composites). Free functions `line_plane_intersection` / `line_cylinder_intersection` are the primitives.

- **EM fields** (`jax/emfield.py`): `EMTensorField` **extends `Volume`** — a field source is also a region of space. Implementors provide `field_strength(point) -> (E, B)` plus the `Volume` methods. `EMTensorField.__call__(tangent_4momentum)` contracts the field with a four-momentum to produce the Lorentz-force change. Fields compose with `+` (`SumField`); concrete sources live in `jax/magnet/` (`solenoid.py`, `loop.py`), `jax/rfcavity/` (`pillbox.py`), with material/absorber in `jax/absorber/`.

- **Particle state** (`jax/kinematics.py`): `ParticleState`/`MuonState` hold `kin: Tangent[Cartesian4]` (the tangent is the four-momentum, scaled by mass) and charge sign `q` (a static field). The key extensibility point is **`scale()`**, which selects the integration independent variable: `MuonStateDct` integrates vs lab time `ct` (scale 1), `MuonStateDz` integrates vs longitudinal position `z`. `build_tangent` rebuilds the state from a derivative PyTree.

- **Integration** (`jax/integrate/`): `propagate.py` glues the above to **diffrax**. `particle_interaction` is the ODE RHS (Lorentz force, respecting `state.scale()`); `diffrax_solve` is an *example* solver (Dopri5 + forward-mode AD adjoint) — expect to write your own per use case. `stepsize.py`'s `BoundaryAwareStepSizeController` wraps a diffrax controller (e.g. `PIDController`) and uses a signed-distance function (from `Volume.signed_distance`) to avoid stepping across field/material boundaries where the RHS is discontinuous.

The data-flow loop is: a `Volume`/`EMTensorField` defines where fields/material exist and how far the nearest boundary is → a `ParticleState` carries the kinematics and chooses the independent variable → diffrax integrates `particle_interaction` with boundary-aware stepping → gradients flow through the whole thing because every piece is an equinox PyTree of JAX arrays.

## Conventions

- jaxtyping aliases (`SFloat`, `Vec3`, `Tangent`, etc.) are **quarantined in `jax/types.py`** so the linter rule for runtime-checked annotations is disabled only there — put new jaxtyping declarations in that file.
- Abstract base classes use `eqx.AbstractVar` for fields and `@abstractmethod`; concrete subclasses declare the real `eqx.field(...)`.
- `references/` holds the physics papers (PDFs) behind the implementations; `notebook/` holds exploratory Jupyter notebooks (lint rules relax `print`/`isort` there). Neither is part of the package.
- Commit messages: you MUST use `Assisted-by:` and not `Co-authored-by:` in the signature of any commit messages
