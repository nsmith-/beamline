"""Export beamline scenes and particle trajectories to Universal Scene Description (USD).

Requires the ``usd-core`` package (``pip install usd-core``).  All USD imports
are deferred so the rest of ``beamline`` remains importable without it.

Unit convention: the stage is configured for millimeters (``metersPerUnit =
0.001``) and Y-up axis. Beamline geometry itself is still authored with z as
the beam axis; the stage's up axis is just a hint for viewer cameras/grids
and does not transform any coordinates.

Typical usage::

    from beamline.jax.export.usd import make_stage, add_volume, add_trajectories

    stage = make_stage("scene.usda")
    add_volume(stage, "/beamline/solenoid", my_solenoid)
    add_volume(stage, "/beamline/cavity", my_cavity)
    add_trajectories(stage, "/trajectories", particle_states)
    stage.Save()
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd


# Display colors (RGB float), one unique color per concrete exportable type.
_COLOR_THIN_SHELL_SOLENOID = (0.2, 0.5, 0.9)  # blue
_COLOR_THICK_SOLENOID = (0.1, 0.8, 0.8)  # cyan
_COLOR_CAVITY = (0.9, 0.5, 0.1)  # orange
_COLOR_ABSORBER = (0.4, 0.7, 0.4)  # green
_COLOR_TRAJECTORY = (1.0, 0.8, 0.0)  # yellow
_COLOR_UNKNOWN = (
    0.6,
    0.6,
    0.6,
)  # gray, fallback for unregistered CylinderVolume subtypes

_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _snake_case(class_name: str) -> str:
    """Convert a ``CamelCase`` class name to a ``snake_case`` prim-name token."""
    return _CAMEL_BOUNDARY_RE.sub("_", class_name).lower()


def _require_pxr() -> None:
    """Raise a clear ImportError when usd-core is not installed."""
    try:
        import pxr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "usd-core is required for USD export.  Install it with:\n"
            "    pip install usd-core\n"
            "or add it as a project dependency:  beamline[usd]"
        ) from exc


def make_stage(path: str) -> Usd.Stage:
    """Create a new USD stage configured for beamline coordinates.

    Sets Y-up axis and millimeter units (``metersPerUnit = 0.001``); z remains
    the beam axis in the authored geometry.

    Args:
        path: File path for the stage (e.g. ``"scene.usda"`` or ``"scene.usd"``).

    Returns:
        A new ``Usd.Stage`` ready to receive prims.
    """
    _require_pxr()
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1e-3)
    return stage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(val) -> float:
    """Convert a JAX scalar or Python number to a plain Python float."""
    return float(np.asarray(val))


def _make_matrix4d(rotation_3x3: np.ndarray, translation_3: np.ndarray):
    """Build a USD Gf.Matrix4d from a 3x3 rotation and 3-vector translation.

    USD Matrix4d is row-major with the translation in the last row::

        | R  0 |
        | t  1 |
    """
    from pxr import Gf

    r = rotation_3x3
    t = translation_3
    return Gf.Matrix4d(
        r[0, 0],
        r[0, 1],
        r[0, 2],
        0.0,
        r[1, 0],
        r[1, 1],
        r[1, 2],
        0.0,
        r[2, 0],
        r[2, 1],
        r[2, 2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    )


def _add_cylinder_prim(
    stage: Usd.Stage,
    prim_path: str,
    radius: float,
    length: float,
    color: tuple[float, float, float],
) -> object:
    """Define a UsdGeom.Cylinder centered at the origin, axis along Z."""
    from pxr import Gf, UsdGeom, Vt

    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.GetAxisAttr().Set("Z")
    cyl.GetRadiusAttr().Set(_to_float(radius))
    cyl.GetHeightAttr().Set(_to_float(length))
    cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return cyl


def _add_tube_prim(
    stage: Usd.Stage,
    prim_path: str,
    inner_radius: float,
    outer_radius: float,
    length: float,
    color: tuple[float, float, float],
    n_segments: int = 32,
) -> object:
    """Define a UsdGeom.Mesh annular tube (hollow cylinder) centered at the
    origin, axis along Z.

    Built from four rings of points (outer/inner, bottom/top) connected into
    quads for the outer wall, bore wall, and top/bottom annular end caps.
    """
    from pxr import Gf, UsdGeom, Vt

    n = n_segments
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    z0, z1 = -length / 2.0, length / 2.0

    def ring(radius: float, z: float) -> list:
        return [
            Gf.Vec3f(radius * c, radius * s, z) for c, s in zip(cos, sin, strict=True)
        ]

    outer_bottom = ring(outer_radius, z0)
    outer_top = ring(outer_radius, z1)
    inner_bottom = ring(inner_radius, z0)
    inner_top = ring(inner_radius, z1)
    points = outer_bottom + outer_top + inner_bottom + inner_top
    ob, ot, ib, it = (0, n, 2 * n, 3 * n)

    face_vertex_counts = []
    face_vertex_indices = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        face_vertex_counts.append(4)
        face_vertex_indices.extend([a, b, c, d])

    for i in range(n):
        j = (i + 1) % n
        # Outer wall: normal points outward (away from the axis).
        quad(ob + i, ob + j, ot + j, ot + i)
        # Bore wall: reversed winding so the normal points inward, into the bore.
        quad(ib + i, it + i, it + j, ib + j)
        # Top annulus (z = z1): normal points up (+z).
        quad(ot + i, ot + j, it + j, it + i)
        # Bottom annulus (z = z0): normal points down (-z).
        quad(ob + i, ib + i, ib + j, ob + j)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return mesh


def _add_xform_prim(stage: Usd.Stage, prim_path: str, transform) -> object:
    """Define a UsdGeom.Xform prim applying a beamline Transform.

    Only the spatial (3x3) block of the rotation and the (x, y, z) part of the
    translation are used — the temporal row/column is not meaningful for geometry.
    """
    from pxr import UsdGeom

    rot3 = np.asarray(transform.rotation[:3, :3])
    trans3 = np.asarray(transform.translation.coords[:3])
    mat = _make_matrix4d(rot3, trans3)

    xf = UsdGeom.Xform.Define(stage, prim_path)
    xf.AddTransformOp().Set(mat)
    return xf


def _add_curve_prim(
    stage: Usd.Stage,
    prim_path: str,
    positions: np.ndarray,
    width: float,
    color: tuple[float, float, float],
) -> object:
    """Define a UsdGeom.BasisCurves polyline from an (N, 3) position array."""
    from pxr import Gf, UsdGeom, Vt

    n = positions.shape[0]
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        )
    )
    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([n]))
    curves.GetWidthsAttr().Set(Vt.FloatArray([float(width)] * n))
    curves.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return curves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_volume(
    stage: Usd.Stage,
    prim_path: str,
    vol: object,
) -> None:
    """Add a Volume, EMTensorField, or MaterialVolume to the stage.

    Dispatches on the concrete type to produce appropriate USD geometry, each
    with its own unique display color:

    - ``ThinShellSolenoid`` → ``Cylinder`` (radius ``R``, height ``L``)
    - ``ThickSolenoid`` → ``Mesh`` annular tube (inner radius ``Rin``, outer
      radius ``Rout``, height ``L``) — a hollow tube, not a solid cylinder
    - Any other ``CylinderVolume`` (e.g. ``PillboxCavity``, ``AbsorberCylinder``)
      → ``Cylinder`` (``radius``, ``length``, per the ABC)
    - ``SumField`` → recurse into ``components``, naming each child prim
      ``<snake_case_type_name>_<index>`` (e.g. two summed ``PillboxCavity``
      instances become ``pillbox_cavity_0`` and ``pillbox_cavity_1``) so
      multiple components of the same or different types remain
      distinguishable in the stage tree
    - ``TransformEMField`` / ``TransformMaterialVolume`` → ``Xform`` parent, with a
      child prim named after the wrapped type's ``<snake_case_type_name>``
      (e.g. wrapping a ``PillboxCavity`` produces a ``pillbox_cavity`` child,
      not a generic ``field``/``material``), so the type stays visible even
      when placed via a transform

    Unknown types produce a ``UserWarning`` and are otherwise skipped. A
    ``CylinderVolume`` subtype with no registered display color also warns
    and falls back to a neutral gray rather than silently reusing another
    type's color.

    Args:
        stage: The target USD stage (from :func:`make_stage`).
        prim_path: Absolute USD prim path, e.g. ``"/beamline/solenoid_0"``.
        vol: A beamline volume / field object.
    """
    _require_pxr()

    from beamline.jax.absorber.volume import AbsorberCylinder, TransformMaterialVolume
    from beamline.jax.emfield import SumField, TransformEMField
    from beamline.jax.geometry import CylinderVolume
    from beamline.jax.magnet.solenoid import ThickSolenoid, ThinShellSolenoid
    from beamline.jax.rfcavity.pillbox import PillboxCavity

    # One unique display color per concrete exportable type (not per category),
    # so e.g. ThinShellSolenoid and ThickSolenoid are visually distinguishable.
    type_colors = {
        ThinShellSolenoid: _COLOR_THIN_SHELL_SOLENOID,
        ThickSolenoid: _COLOR_THICK_SOLENOID,
        PillboxCavity: _COLOR_CAVITY,
        AbsorberCylinder: _COLOR_ABSORBER,
    }

    if isinstance(vol, SumField):
        for i, comp in enumerate(vol.components):
            name = _snake_case(type(comp).__name__)
            add_volume(stage, f"{prim_path}/{name}_{i}", comp)

    elif isinstance(vol, TransformEMField):
        _add_xform_prim(stage, prim_path, vol.transform)
        name = _snake_case(type(vol.field).__name__)
        add_volume(stage, f"{prim_path}/{name}", vol.field)

    elif isinstance(vol, TransformMaterialVolume):
        _add_xform_prim(stage, prim_path, vol.transform)
        name = _snake_case(type(vol.material).__name__)
        add_volume(stage, f"{prim_path}/{name}", vol.material)

    elif isinstance(vol, ThinShellSolenoid):
        _add_cylinder_prim(
            stage, prim_path, vol.R, vol.L, type_colors[ThinShellSolenoid]
        )

    elif isinstance(vol, ThickSolenoid):
        _add_tube_prim(
            stage, prim_path, vol.Rin, vol.Rout, vol.L, type_colors[ThickSolenoid]
        )

    elif isinstance(vol, CylinderVolume):
        # radius/length come from the ABC, shared by e.g. PillboxCavity and
        # AbsorberCylinder; the display color is looked up per concrete type,
        # falling back to a neutral gray (with a warning) for anything not
        # explicitly registered above.
        color = type_colors.get(type(vol))
        if color is None:
            warnings.warn(
                f"add_volume: {type(vol).__name__!r} at {prim_path!r} has no "
                "registered display color — using a fallback gray",
                stacklevel=2,
            )
            color = _COLOR_UNKNOWN
        _add_cylinder_prim(stage, prim_path, vol.radius, vol.length, color)

    else:
        warnings.warn(
            f"add_volume: unknown type {type(vol).__name__!r} at {prim_path!r} — skipping",
            stacklevel=2,
        )


def add_trajectories(
    stage: Usd.Stage,
    prim_path: str,
    states: object,
    *,
    width: float = 2.0,
) -> None:
    """Add particle trajectories as UsdGeom.BasisCurves polylines.

    Each trajectory is a separate ``BasisCurves`` prim under ``prim_path``.
    Positions are taken from ``states.kin.p`` (a ``Cartesian4`` array); only
    the spatial (x, y, z) components are used.

    Args:
        stage: The target USD stage.
        prim_path: Base prim path, e.g. ``"/trajectories"``.
        states: A ``ParticleState`` (or batch) returned by a diffrax solver.
            Leading axis is time; an optional second axis is the particle batch.
        width: Curve display width in mm (rendered as a tube by most viewers).
    """
    _require_pxr()

    # coords shape: (n_times, 4) or (n_times, n_particles, 4)
    positions = np.asarray(states.kin.p.coords)[..., :3]  # (..., 3)

    if positions.ndim == 2:
        _add_curve_prim(
            stage, f"{prim_path}/particle_0", positions, width, _COLOR_TRAJECTORY
        )
    elif positions.ndim == 3:
        for i in range(positions.shape[1]):
            _add_curve_prim(
                stage,
                f"{prim_path}/particle_{i}",
                positions[:, i, :],
                width,
                _COLOR_TRAJECTORY,
            )
    else:
        raise ValueError(
            f"Expected positions of shape (n_times, 3) or (n_times, n_particles, 3), "
            f"got {positions.shape}"
        )
