"""Helpers for writing particle states to files"""

from collections.abc import Sequence
from pathlib import Path

import hepunits as u
import numpy as np

from beamline.jax.kinematics import ParticleState


def write_states_csv(
    path: Path,
    states: ParticleState,
    *,
    index_names: Sequence[str] = (),
    suffix: str = "",
) -> None:
    """Write a batch of particle states to CSV, one row per state

    Args:
        path: Output file
        states: A ParticleState, with any number of leading batch dimensions
        index_names: If given, one name per batch dimension; the index along each
            is written as a leading integer column
        suffix: Appended to each kinematic column name (e.g. "f" for final states)
    """
    pos = np.asarray(states.kin.p.coords) / u.mm
    mom = np.asarray(states.kin.t.coords) / u.MeV
    batch_shape = pos.shape[:-1]
    if index_names and len(index_names) != len(batch_shape):
        msg = f"Expected {len(batch_shape)} index names, got {len(index_names)}"
        raise ValueError(msg)
    columns = [
        np.indices(batch_shape).reshape(len(batch_shape), -1).T[:, i]
        for i in range(len(index_names))
    ]
    columns += [pos.reshape(-1, 4)[:, i] for i in range(4)]
    columns += [mom.reshape(-1, 4)[:, i] for i in range(4)]
    names = ["x", "y", "z", "t", "px", "py", "pz", "E"]
    header = ",".join([*index_names, *(name + suffix for name in names)])
    fmt = ["%d"] * len(index_names) + ["%.6f"] * len(names)
    np.savetxt(
        path,
        np.stack(columns, axis=-1),
        fmt=fmt,
        delimiter=",",
        header=header,
        comments="",
    )
