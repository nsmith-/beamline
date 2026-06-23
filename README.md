# beamline

This is an R&D package to develop differentiable muon beamline simulations in
jax, tailored towards the needs of muon cooling channel design optimization, in
the context of designing a [muon collider](https://www.muoncollider.us/).

## Introduction

The motivation for a ground-up simulation is three-fold. First, it is driven by
limitations of existing tooling, as muon cooling is at a particularly
interesting intersection of multiphysics simulation needs: beam transport
optics, material interaction, and collective beam dynamics all are expected to
play a non-trivial role.  Second, it is desirable to leverage a computational
framework with first-class automatic differentiation support and GPU
acceleration "for free", such as jax. A third motivation is simply that this is
a fun problem to work on and understand.

Prior art in the area of muon beamline simulations includes:
- [iCOOL](https://www.cap.bnl.gov/ICOOL/fernow/v330/)
- [G4Beamline](https://www.muonsinc.com/Website1/G4beamline)
- [BDSim](https://bdsim-collaboration.github.io/web/)
- [RF-Track](https://abpcomputing.web.cern.ch/codes/codes_pages/RF-Track/)

General beam simulations that have considered automatic differentiation include:
- [Synergia](https://synergia.fnal.gov/)
- [Cheetah](https://cheetah-accelerator.readthedocs.io/en/latest/)
- [WarpX](https://warpx.readthedocs.io/en/24.05/index.html)

and likely several others.

The key modeling needs for us are: structures for external electromagnetic
fields such as solenoids, RF cavities, multipole magnets; volumes of absorber
material, simulating energy loss including multiple scattering and energy
straggling effects; and beam propagation, including collective effects such as
beam-beam interaction, beam loading, and wake fields as the beam passes through
material interfaces.

## Features

We are in the early days :)

## Getting Started

This package is pip-installable. We recommend using
[uv](https://docs.astral.sh/uv/) for virtual environment management. With it,
you can set up the environment and install dependencies using:

```bash
uv sync
```

All results are curated through the test infrastructure while we are in development
mode. The eventual goal is a library and command-line interface. To produce all test
artifacts, you can run:
```bash
uv run pytest
```
which will produce a `test_artifacts` directory containing figures and data
files from the test results. This may take a few minutes.
