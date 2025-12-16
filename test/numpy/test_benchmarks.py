"""Cooling code benchmark tests.

As described in:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf

"""

import gzip
import pickle
from functools import partial
from typing import Any

import hepunits as u
import numpy as np
import pytest
from scipy.special import jn_zeros

from beamline.numpy.integrate import solve_ivp
from beamline.numpy.kinematics import (
    ParticleState,
    make_muon,
    ode_tangent_dz,
)
from beamline.numpy.pillbox import PillboxCavity
from beamline.numpy.solenoid import ThickSolenoid


@pytest.fixture(scope="module")
def output_dict():
    output: dict[str, Any] = {}
    yield output
    with gzip.open("benchmark_results.pkl.gz", "wb") as fout:
        pickle.dump(output, fout)


def test_benchmark_3p2_solenoid(output_dict):
    """Benchmark 3.2: Muon through solenoid

    Parameters from Table 2
    """
    solenoid = ThickSolenoid(
        Rin=250.0 * u.mm,
        Rout=419.3 * u.mm,
        jphi=500.0 * u.A / u.mm**2,
        L=140.0 * u.mm,
    )
    out = []
    for xpos in np.arange(-200.0 * u.mm, 201.0 * u.mm, 10.0 * u.mm):
        start = make_muon(
            z=-500.0 * u.mm,
            pz=200.0 * u.MeV / u.c_light,
            x=xpos,
            charge=1,
        )
        sol = solve_ivp(
            fun=partial(ode_tangent_dz, solenoid),
            y0=start,
            t_span=(-500.0 * u.mm, 500.0 * u.mm),
            rtol=1e-4,
        )
        end: ParticleState = sol.y[-1]
        out.append((start, end))
    output_dict["benchmark_3p2_solenoid"] = out


def test_benchmark_3p3_rf_cell(output_dict):
    """Benchmark 3.3: Muon through RF cavity

    Parameters from Table 3
    """
    v01 = jn_zeros(0, 1)[0]
    frequency = 704.0 * u.MHz
    # time to get to 0 position is 500mm / beta*c
    mu0 = make_muon(pz=200.0 * u.MeV / u.c_light)
    t0 = 500 * u.mm / (mu0.momentum.beta * u.c_light)
    # print(f"Expecting ct0 = {u.c_light*t0} ns")
    # advance by pi/2 so we are in bunching mode
    # i.e. refence particle momentum is unchanged
    phase = -2 * u.pi * ((t0 * frequency - 0.25) % 1.0)

    cavity = PillboxCavity(
        length=183.6 * u.mm,
        radius=v01 * u.c_light / (2 * u.pi * frequency),
        E0=30.0 * u.MV / u.m,
        mode="TM",
        m=0,
        n=1,
        p=0,
        phase=phase,
    )
    assert cavity.frequency == pytest.approx(frequency, rel=1e-12)

    out = []
    for ix in range(21):
        xpos = ix * 10.0 * u.mm
        for it in range(11):
            tpos = it * 0.1 / 0.704 * u.ns
            start = make_muon(
                z=-500.0 * u.mm,
                pz=200.0 * u.MeV / u.c_light,
                x=xpos,
                ct=tpos * u.c_light,
                charge=1,
            )
            # optionally skip forward to near cavity entrance
            # start.position += 400.0 * u.mm * start.momentum / start.momentum.pz
            sol = solve_ivp(
                fun=partial(ode_tangent_dz, cavity),
                y0=start,
                t_span=(start.position.z, 500.0 * u.mm),
                t_eval=[0., 500.0 * u.mm],
                rtol=1e-4,
                max_step=10 * u.mm,
            )
            end: ParticleState = sol.y[-1]
            out.append((start, end))
    output_dict["benchmark_3p3_rf_cell"] = out
