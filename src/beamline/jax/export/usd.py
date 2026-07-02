"""Export beamline scenes and particle trajectories to Universal Scene Description (USD).

Requires the ``usd-core`` package (``pip install usd-core``).  All USD imports
are deferred so the rest of ``beamline`` remains importable without it.

Unit convention: the stage is configured for millimeters (``metersPerUnit =
0.001``) and Y-up axis. Beamline geometry itself is still authored with z as
the beam axis; the stage's up axis is just a hint for viewer cameras/grids
and does not transform any coordinates.

Cylinder/tube volumes carry a ``st`` UV primvar and are bound to a
``UsdPreviewSurface`` material textured with a small generated color-canvas
PNG (one per concrete type, reused across instances). Requires ``pillow``
(part of the ``usd`` extra) in addition to ``usd-core``.

Scenes are packaged as self-contained ``.usdz`` files: :func:`make_stage`
authors the stage and its texture PNGs into a private temporary working
directory, and :func:`save_usdz` flattens everything referenced by the
stage into a single ``.usdz`` at the requested path and removes the working
directory — so there's one portable file to move or share, not a ``.usda``
plus a loose ``textures/`` folder that has to travel with it.

Typical usage::

    from beamline.jax.export.usd import make_stage, add_volume, add_trajectories, save_usdz

    stage = make_stage("scene.usdz")
    add_volume(stage, "/beamline/solenoid", my_solenoid)
    add_volume(stage, "/beamline/cavity", my_cavity)
    add_trajectories(stage, "/trajectories", particle_states)
    save_usdz(stage)
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
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

# Slightly translucent so a marker/light traveling through the interior of a
# solenoid/cavity/absorber is still visible from outside.
_VOLUME_OPACITY = 0.85

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
    """Create a new USD stage for building a self-contained ``.usdz`` package.

    ``path`` is the desired final path of the exported package — the stage
    itself and any assets it references (e.g. the generated color-canvas
    textures from :func:`add_volume`) are authored into a private temporary
    working directory, not ``path`` directly. Call :func:`save_usdz` once the
    scene is built to flatten everything into a single ``.usdz`` file at
    ``path`` and remove the working directory.

    Sets Y-up axis and millimeter units (``metersPerUnit = 0.001``); z remains
    the beam axis in the authored geometry.

    Args:
        path: Desired final path for the package, e.g. ``"scene.usdz"``.

    Returns:
        A new ``Usd.Stage`` ready to receive prims.
    """
    _require_pxr()
    from pxr import Usd, UsdGeom

    if not path.endswith(".usdz"):
        raise ValueError(f"make_stage: path must end in '.usdz', got {path!r}")

    workdir = tempfile.mkdtemp(prefix="beamline_usdz_")
    stem = os.path.splitext(os.path.basename(path))[0]
    stage = Usd.Stage.CreateNew(os.path.join(workdir, f"{stem}.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1e-3)

    stage._beamline_usdz_path = os.path.abspath(path)
    stage._beamline_workdir = workdir
    return stage


def save_usdz(stage: Usd.Stage) -> None:
    """Flush ``stage`` and package it, plus every asset it references (e.g.
    generated color-canvas textures — see :func:`add_volume`), into the
    single self-contained ``.usdz`` file requested via :func:`make_stage`.

    Removes the temporary working directory :func:`make_stage` created
    ``stage`` in, regardless of success, so call this exactly once per stage.

    Args:
        stage: A stage created by :func:`make_stage` (not saved yet).
    """
    _require_pxr()
    from pxr import Sdf, UsdUtils

    try:
        usdz_path = stage._beamline_usdz_path
        workdir = stage._beamline_workdir
    except AttributeError as exc:
        raise ValueError(
            "save_usdz: stage was not created by make_stage() — no known "
            "working directory/target .usdz path to package"
        ) from exc

    try:
        root_layer = stage.GetRootLayer()
        root_layer.Save()
        ok = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(root_layer.realPath), usdz_path
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if not ok:
        raise RuntimeError(f"save_usdz: failed to create USDZ package at {usdz_path!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(val) -> float:
    """Convert a JAX scalar or Python number to a plain Python float."""
    return float(np.asarray(val))


Transform3 = tuple[np.ndarray, np.ndarray]  # (3x3 rotation, 3-vector translation)


def _compose_transform(outer: Transform3 | None, transform: object) -> Transform3:
    """Compose an already-accumulated global ``(rotation, translation)`` with
    one more beamline ``Transform`` found while unwrapping a nested
    ``TransformEMField``/``TransformMaterialVolume``.

    ``transform`` maps its wrapped object's local coordinates into the frame
    that ``outer`` already maps to global (or straight to global if ``outer``
    is ``None``): ``global = outer.rotation @ (transform.rotation @ x +
    transform.translation) + outer.translation``.
    """
    r = np.asarray(transform.rotation[:3, :3])
    t = np.asarray(transform.translation.coords[:3])
    if outer is None:
        return r, t
    rot, trans = outer
    return rot @ r, rot @ t + trans


def _apply_transform(points: np.ndarray, transform: Transform3 | None) -> np.ndarray:
    """Map an (N, 3) local point cloud into global coordinates."""
    if transform is None:
        return points
    rot, trans = transform
    return points @ rot.T + trans


def _innermost_type(vol: object) -> type:
    """The concrete type of ``vol`` after unwrapping any
    ``TransformEMField``/``TransformMaterialVolume`` layers, for naming a
    transformed component after what it actually is rather than the wrapper.
    """
    from beamline.jax.absorber.volume import TransformMaterialVolume
    from beamline.jax.emfield import TransformEMField

    while isinstance(vol, TransformEMField | TransformMaterialVolume):
        vol = vol.field if isinstance(vol, TransformEMField) else vol.material
    return type(vol)


def _texture_dir(stage: Usd.Stage) -> str:
    """Directory next to the stage's root layer file where generated
    textures are written, so the relative asset paths referencing them
    resolve correctly."""
    return os.path.join(os.path.dirname(stage.GetRootLayer().realPath), "textures")


def _make_canvas_texture(
    path: str,
    label: str,
    color: tuple[float, float, float],
    size: int = 256,
    width: int | None = None,
    squeeze: float = 1.0,
    font_scale: float = 1.0,
) -> None:
    """Generate a small procedural color-canvas PNG for a concrete
    exportable type: its flat display color as a base, overlaid with a
    UV-visualizing grid and the type name, so textured viewers show more
    than a flat swatch and any UV-mapping mistakes (stretching, seams) are
    easy to spot.

    ``width`` (defaulting to ``size``, i.e. a square canvas) lets the canvas
    be wider than it is tall, for surfaces whose ``u`` UV axis is unwrapped
    across a much longer physical distance than ``v`` (e.g. a solenoid's
    outer wall, unwrapped around its circumference against its length) —
    see the ``width`` argument to :func:`_bind_canvas_material`. That alone
    gives more horizontal pixels to work with (so the grid/label don't
    blur), but doesn't by itself undo any stretch — the label is additionally
    rendered at its natural proportions and then squeezed narrower with an
    affine resize by ``squeeze`` (< 1) before being composited, so it arrives
    back at (roughly) its natural aspect once the UV mapping re-stretches it.
    ``squeeze`` is independent of ``width``/``size`` — callers with a fixed
    resolution budget (so the canvas can't be widened all the way to the
    physical circumference:length ratio) still pass the remaining ratio
    here.

    ``font_scale`` scales the label up (or down) from its default size
    (``size // 13``) — ``v`` is never stretched by the UV mapping (unlike
    ``u``), so a face whose ``v`` axis covers a physically short distance
    (e.g. a thick solenoid's outer wall, wrapped around a length much
    shorter than its circumference) can still end up with a tiny label in
    absolute terms even though its proportions are correct; callers pass a
    larger ``font_scale`` to compensate.
    """
    from PIL import Image, ImageDraw, ImageFont

    width = size if width is None else width
    height = size
    rgb = tuple(int(round(c * 255)) for c in color)
    img = Image.new("RGB", (width, height), rgb)
    draw = ImageDraw.Draw(img)

    grid_rgb = tuple(min(255, c + 40) for c in rgb)
    n_lines = 8
    for k in range(1, n_lines):
        px = width * k // n_lines
        draw.line([(px, 0), (px, height)], fill=grid_rgb, width=1)
    for k in range(1, n_lines):
        py = height * k // n_lines
        draw.line([(0, py), (width, py)], fill=grid_rgb, width=1)

    text = label.replace("_", " ")
    font = ImageFont.load_default(size=round(height / 13 * font_scale))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Render the label onto its own canvas at natural proportions, then
    # squeeze it horizontally so the glyphs arrive back at (roughly) their
    # natural aspect once the UV mapping re-stretches them across the
    # geometry.
    text_layer = Image.new("RGBA", (max(tw, 1), max(th, 1)), (0, 0, 0, 0))
    ImageDraw.Draw(text_layer).text(
        (-bbox[0], -bbox[1]), text, fill=(255, 255, 255, 255), font=font
    )
    narrow_w = max(1, round(tw * squeeze))
    text_layer = text_layer.resize((narrow_w, th), Image.LANCZOS)
    img.paste(
        text_layer,
        (round((width - narrow_w) / 2), round((height - th) / 2)),
        text_layer,
    )

    img.save(path)


def _bind_canvas_material(
    stage: Usd.Stage,
    mesh: object,
    label: str,
    color: tuple[float, float, float],
    texture_width: int | None = None,
    texture_squeeze: float = 1.0,
    texture_font_scale: float = 1.0,
    opacity: float = 1.0,
) -> None:
    """Create (once per ``label``) a ``UsdPreviewSurface`` material textured
    with a generated color-canvas PNG (:func:`_make_canvas_texture`) sampled
    through the mesh's ``st`` UV primvar, and bind it to ``mesh``.

    The Material prim and its texture file are cached per ``label`` on the
    stage, so e.g. several ``PillboxCavity`` instances in a ``SumField``
    share one generated texture and Material rather than duplicating both
    per instance.

    ``texture_width`` widens the generated canvas beyond the default square
    (see :func:`_make_canvas_texture`) — pass a value greater than the
    default texture size for meshes whose ``u`` UV axis wraps a much longer
    physical distance than ``v`` (a cylindrical wall unwrapped around its
    circumference), so the grid/label stay crisp instead of blurring across
    a few stretched texels. ``texture_squeeze`` additionally pre-compresses
    the label text (see :func:`_make_canvas_texture`) for whatever fraction
    of that circumference:length stretch ``texture_width`` doesn't already
    cover on its own. ``texture_font_scale`` enlarges the label itself, for
    faces whose ``v`` axis spans a physically short distance.

    ``opacity`` (< 1) makes the material translucent, e.g. so a marker/light
    traveling through the volume's interior stays visible from outside.
    """
    from pxr import Sdf, UsdShade

    material_path = f"/Materials/{label}"
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid():
        material = UsdShade.Material(existing)
    else:
        tex_dir = _texture_dir(stage)
        os.makedirs(tex_dir, exist_ok=True)
        _make_canvas_texture(
            os.path.join(tex_dir, f"{label}.png"),
            label,
            color,
            width=texture_width,
            squeeze=texture_squeeze,
            font_scale=texture_font_scale,
        )

        material = UsdShade.Material.Define(stage, material_path)

        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        tex_sampler = UsdShade.Shader.Define(stage, f"{material_path}/diffuseTexture")
        tex_sampler.CreateIdAttr("UsdUVTexture")
        tex_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            f"./textures/{label}.png"
        )
        tex_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )
        tex_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        pbr_shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
        pbr_shader.CreateIdAttr("UsdPreviewSurface")
        pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbr_shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).ConnectToSource(tex_sampler.ConnectableAPI(), "rgb")
        pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(
            pbr_shader.ConnectableAPI(), "surface"
        )

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh).Bind(material)


def _add_cylinder_prim(
    stage: Usd.Stage,
    prim_path: str,
    radius: float,
    length: float,
    color: tuple[float, float, float],
    label: str,
    n_segments: int = 64,
    transform: Transform3 | None = None,
) -> object:
    """Define a solid triangulated cylinder mesh, axis along Z.

    Built as an explicit ``Mesh`` (side wall + two end caps) rather than
    USD's implicit ``UsdGeom.Cylinder`` schema, which has no poly-count
    control at all — it's a quadric each viewer tessellates at its own
    resolution (e.g. usdview's viewport "complexity" setting). Authoring it
    as a mesh with a fixed ``n_segments`` gives consistent poly density
    across every viewer, matching :func:`_add_tube_prim`.

    If ``transform`` is given, the local (axis-along-Z, origin-centered)
    point cloud is mapped into global coordinates before being written, so
    the prim carries no transform op of its own — see :func:`add_volume`.

    Also authors a ``faceVarying`` ``st`` UV primvar (side wall unwrapped
    cylindrically, end caps mapped as a planar disc) and binds a generated
    color-canvas texture material keyed by ``label`` — see
    :func:`_bind_canvas_material`.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt

    radius, length = _to_float(radius), _to_float(length)
    n = n_segments
    # Offset by -pi/2 so the wraparound seam (angle index n-1 back to 0)
    # sits on the -y axis rather than +x.
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) - np.pi / 2.0
    cos, sin = np.cos(angles), np.sin(angles)
    z0, z1 = -length / 2.0, length / 2.0

    def ring(z: float) -> np.ndarray:
        return np.stack([radius * cos, radius * sin, np.full(n, z)], axis=-1)

    points = np.concatenate([ring(z0), ring(z1), [[0.0, 0.0, z0]], [[0.0, 0.0, z1]]])
    points = _apply_transform(points, transform)
    points = [Gf.Vec3f(*(float(v) for v in p)) for p in points]
    b, t, bc, tc = 0, n, 2 * n, 2 * n + 1

    # Planar disc UV for the end caps: ring point i sits at
    # (0.5 + 0.5*cos, 0.5 + 0.5*sin), the center prim at (0.5, 0.5).
    disc_u = 0.5 + 0.5 * cos
    disc_v = 0.5 + 0.5 * sin
    center_uv = (0.5, 0.5)

    face_vertex_counts = []
    face_vertex_indices = []
    uvs: list[tuple[float, float]] = []

    def quad(a: int, b_: int, c: int, d: int, uv: tuple) -> None:
        face_vertex_counts.append(4)
        face_vertex_indices.extend([a, b_, c, d])
        uvs.extend(uv)

    def tri(a: int, b_: int, c: int, uv: tuple) -> None:
        face_vertex_counts.append(3)
        face_vertex_indices.extend([a, b_, c])
        uvs.extend(uv)

    for i in range(n):
        j = (i + 1) % n
        u0, u1 = i / n, (i + 1) / n
        uv_i = (disc_u[i], disc_v[i])
        uv_j = (disc_u[j], disc_v[j])
        # Side wall: normal points outward (away from the axis).
        quad(
            b + i,
            b + j,
            t + j,
            t + i,
            [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)],
        )
        # Bottom cap (z = z0): normal points down (-z).
        tri(bc, b + j, b + i, [center_uv, uv_j, uv_i])
        # Top cap (z = z1): normal points up (+z).
        tri(tc, t + i, t + j, [center_uv, uv_i, uv_j])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([_VOLUME_OPACITY]))

    st_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st_primvar.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))

    _bind_canvas_material(stage, mesh, label, color, opacity=_VOLUME_OPACITY)
    return mesh


def _add_tube_prim(
    stage: Usd.Stage,
    prim_path: str,
    inner_radius: float,
    outer_radius: float,
    length: float,
    color: tuple[float, float, float],
    label: str,
    n_segments: int = 64,
    transform: Transform3 | None = None,
) -> object:
    """Define a UsdGeom.Mesh annular tube (hollow cylinder), axis along Z.

    Built from four rings of points (outer/inner, bottom/top) connected into
    quads for the outer wall, bore wall, and top/bottom annular end caps.

    If ``transform`` is given, the local (axis-along-Z, origin-centered)
    point cloud is mapped into global coordinates before being written, so
    the prim carries no transform op of its own — see :func:`add_volume`.

    Also authors a ``faceVarying`` ``st`` UV primvar (walls unwrapped
    cylindrically, annular caps mapped radially: ``u`` = angle, ``v`` = 0 at
    the bore and 1 at the outer wall) and binds a generated color-canvas
    texture material keyed by ``label`` — see :func:`_bind_canvas_material`.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt

    n = n_segments
    # Offset by -pi/2 so the wraparound seam (angle index n-1 back to 0)
    # sits on the -y axis rather than +x.
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) - np.pi / 2.0
    cos, sin = np.cos(angles), np.sin(angles)
    z0, z1 = -length / 2.0, length / 2.0

    def ring(radius: float, z: float) -> np.ndarray:
        return np.stack([radius * cos, radius * sin, np.full(n, z)], axis=-1)

    points = np.concatenate(
        [
            ring(outer_radius, z0),
            ring(outer_radius, z1),
            ring(inner_radius, z0),
            ring(inner_radius, z1),
        ]
    )
    points = _apply_transform(points, transform)
    points = [Gf.Vec3f(*(float(v) for v in p)) for p in points]
    ob, ot, ib, it = (0, n, 2 * n, 3 * n)

    face_vertex_counts = []
    face_vertex_indices = []
    uvs: list[tuple[float, float]] = []

    def quad(a: int, b: int, c: int, d: int, uv: tuple) -> None:
        face_vertex_counts.append(4)
        face_vertex_indices.extend([a, b, c, d])
        uvs.extend(uv)

    for i in range(n):
        j = (i + 1) % n
        u0, u1 = i / n, (i + 1) / n
        # Outer wall: normal points outward (away from the axis).
        quad(
            ob + i,
            ob + j,
            ot + j,
            ot + i,
            [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)],
        )
        # Bore wall: reversed winding so the normal points inward, into the bore.
        quad(
            ib + i,
            it + i,
            it + j,
            ib + j,
            [(u0, 0.0), (u0, 1.0), (u1, 1.0), (u1, 0.0)],
        )
        # Top annulus (z = z1): normal points up (+z); v = 1 at outer, 0 at
        # inner. u is mirrored (1 - u) relative to the other three faces:
        # this face is seen from the opposite side (+z looking toward -z,
        # vs. the outer/bore walls and bottom cap which are all seen from
        # outside looking toward increasing angle in the same sense as u
        # increases), so an unmirrored u would read the label backwards.
        quad(
            ot + i,
            ot + j,
            it + j,
            it + i,
            [(1.0 - u0, 1.0), (1.0 - u1, 1.0), (1.0 - u1, 0.0), (1.0 - u0, 0.0)],
        )
        # Bottom annulus (z = z0): normal points down (-z).
        quad(
            ob + i,
            ib + i,
            ib + j,
            ob + j,
            [(u0, 1.0), (u0, 0.0), (u1, 0.0), (u1, 1.0)],
        )

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([_VOLUME_OPACITY]))

    st_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st_primvar.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))

    # The outer wall's u axis is unwrapped around the full circumference
    # while v only spans the (typically much shorter) length, so a square
    # texture would stretch its label wide around the annulus. Widening the
    # canvas so its pixel aspect matches that circumference:length ratio
    # undoes the stretch on its own (a texel then covers the same physical
    # distance along u as along v) — capped so a very thin/wide tube
    # doesn't demand an enormous texture, with ``texture_squeeze`` making up
    # whatever fraction of the ratio the cap left uncompensated.
    aspect = (2.0 * np.pi * outer_radius) / length
    capped_aspect = np.clip(aspect, 1.0, 8.0)
    _bind_canvas_material(
        stage,
        mesh,
        label,
        color,
        texture_width=round(256 * capped_aspect),
        texture_squeeze=float(capped_aspect / aspect),
        texture_font_scale=2.0,
        opacity=_VOLUME_OPACITY,
    )
    return mesh


def _add_curve_prim(
    stage: Usd.Stage,
    prim_path: str,
    positions: np.ndarray,
    width: float,
    color: tuple[float, float, float],
    opacity: float = 1.0,
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
    curves.CreateDisplayOpacityAttr().Set(Vt.FloatArray([float(opacity)]))

    # `extent` is a required property on boundable prims (UsdGeomBoundable).
    # usdview computes it on the fly via its own bbox cache, but most other
    # viewers (Blender's USD importer, Quick Look, real-time Hydra viewers,
    # etc.) rely on the *authored* value for culling and silently drop
    # geometry that's missing one.
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    pad = float(width) / 2.0
    curves.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(v) for v in (mins - pad))),
                Gf.Vec3f(*(float(v) for v in (maxs + pad))),
            ]
        )
    )
    return curves


def _blackbody_rgb(color_temperature: float) -> tuple[float, float, float]:
    """Blackbody color at ``color_temperature`` (Kelvin), normalized so its
    brightest channel is 1.0 — ``UsdLux.BlackbodyTemperatureAsRgb`` returns
    unnormalized radiance ratios (e.g. > 1 in the red channel for warm
    temperatures) intended for a light's ``inputs:color``, which is fine
    multiplied against ``inputs:intensity`` but would clip/wash out if used
    directly as a plain material color.
    """
    from pxr import UsdLux

    r, g, b = UsdLux.BlackbodyTemperatureAsRgb(float(color_temperature))
    peak = max(r, g, b, 1e-6)
    return (r / peak, g / peak, b / peak)


def _bind_glow_material(
    stage: Usd.Stage,
    prim: object,
    color: tuple[float, float, float],
    opacity: float,
    label: str,
    emissive_scale: float = 1.0,
) -> None:
    """Create (once per ``label``) an emissive, translucent
    ``UsdPreviewSurface`` material and bind it to ``prim``, analogous to
    :func:`_bind_canvas_material` but a flat emissive color rather than a
    textured one — cached so e.g. every particle's marker shell in one
    :func:`add_trajectories` call shares a single material.

    ``emissive_scale`` multiplies ``color`` for ``emissiveColor`` only
    (``diffuseColor`` stays at ``color``, i.e. <= 1 per channel) — pushing it
    above 1 gives HDR-aware renderers (Storm, Blender/Filament, and to some
    extent RealityKit) something to bloom, which is otherwise the only way
    to get a soft glow out of USD: there's no bloom/post-process schema, so
    :func:`_add_moving_glow_sphere_prim` fakes it with concentric shells
    instead, and a bit of extra emissive punch on the outer ones helps them
    still read as bright despite their low opacity.
    """
    from pxr import Gf, Sdf, UsdShade

    material_path = f"/Materials/{label}"
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid():
        material = UsdShade.Material(existing)
    else:
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(c * emissive_scale for c in color))
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim).Bind(material)


# Concentric glow shells, relative to the marker's nominal radius/opacity:
# a bright near-opaque core plus two larger, more transparent "halo" shells
# with progressively boosted emissive intensity, so the fade-out still reads
# as glowing rather than just fading to gray.
# (radius_mult, opacity_mult, emissive_scale, name)
_GLOW_SHELLS = (
    (1.0, 1.0, 1.5, "core"),
    (2.0, 0.35, 2.5, "halo0"),
    (3.5, 0.12, 4.0, "halo1"),
)


def _add_moving_glow_sphere_prim(
    stage: Usd.Stage,
    prim_path: str,
    positions: np.ndarray,
    color_temperature: float,
    radius: float,
    opacity: float,
) -> object:
    """Define a small ``Xform`` group, position time-sampled to travel along
    ``positions`` (one time code per sample, time code ``i`` ↔
    ``positions[i]``), wrapping concentric emissive/translucent
    ``UsdGeom.Sphere`` shells (see :data:`_GLOW_SHELLS`) that together fake a
    soft "fairy light" glow — a visible stand-in for a moving point light.

    A ``UsdLux.SphereLight`` is invisible in Quick Look/AR Quick Look (the
    ``.usdz`` viewer behind macOS Preview.app): its ARKit-compatible prim
    allowlist excludes every ``UsdLux`` type outright, so any light prim is
    silently ignored. Actual ``Mesh``/``Sphere`` prims with emissive
    materials render everywhere, including there — and grouping them under
    an ``Xform`` (rather than nesting them in each other) keeps every shell a
    sibling Gprim, satisfying ARKit's rule against nesting one Gprim inside
    another.
    """
    from pxr import Gf, Usd, UsdGeom, Vt

    color = _blackbody_rgb(color_temperature)
    group = UsdGeom.Xform.Define(stage, prim_path)

    for radius_mult, opacity_mult, emissive_scale, name in _GLOW_SHELLS:
        shell_radius = radius * radius_mult
        shell_opacity = opacity * opacity_mult
        sphere = UsdGeom.Sphere.Define(stage, f"{prim_path}/{name}")
        sphere.CreateRadiusAttr().Set(float(shell_radius))
        sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        sphere.CreateDisplayOpacityAttr().Set(Vt.FloatArray([float(shell_opacity)]))

        label = (
            f"glow_{name}_{round(color_temperature)}k_op{round(shell_opacity * 100)}"
        )
        _bind_glow_material(
            stage, sphere, color, shell_opacity, label, emissive_scale=emissive_scale
        )

    translate_op = UsdGeom.Xformable(group).AddTranslateOp()
    for i, p in enumerate(positions):
        translate_op.Set(
            Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])), Usd.TimeCode(i)
        )
    return group


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_volume(
    stage: Usd.Stage,
    prim_path: str,
    vol: object,
    *,
    transform: Transform3 | None = None,
) -> None:
    """Add a Volume, EMTensorField, or MaterialVolume to the stage.

    Dispatches on the concrete type to produce appropriate USD geometry, each
    with its own unique display color:

    - ``ThinShellSolenoid`` → ``Mesh`` cylinder (radius ``R``, height ``L``)
    - ``ThickSolenoid`` → ``Mesh`` annular tube (inner radius ``Rin``, outer
      radius ``Rout``, height ``L``) — a hollow tube, not a solid cylinder
    - Any other ``CylinderVolume`` (e.g. ``PillboxCavity``, ``AbsorberCylinder``)
      → ``Mesh`` cylinder (``radius``, ``length``, per the ABC)
    - ``SumField`` → recurse into ``components``, naming each child prim
      ``<snake_case_type_name>_<index>`` (e.g. two summed ``PillboxCavity``
      instances become ``pillbox_cavity_0`` and ``pillbox_cavity_1``); the
      type name is the *innermost* wrapped type, so a transformed component
      still gets a meaningful label instead of e.g. ``transform_em_field_0``
    - ``TransformEMField`` / ``TransformMaterialVolume`` → no ``Xform`` prim is
      emitted. The rotation/translation is composed with any pending
      transform from an enclosing wrapper and baked directly into the
      wrapped geometry's vertex positions, so every prim this function
      creates ends up as a single mesh already in global (stage)
      coordinates, with no transform op of its own.

      This matters because several real-world viewers/importers (notably
      Blender's USD importer) name objects from the leaf prim only, not its
      full USD path — if that leaf name were reused across sibling ``Xform``
      parents (as it was when this wrapped a separate parent/child pair),
      those importers silently collapse the distinguishing parent path and
      you're left with colliding names like ``pillbox_cavity`` /
      ``pillbox_cavity.001``. Baking the transform into one uniquely-named
      prim avoids that entirely.

    Unknown types produce a ``UserWarning`` and are otherwise skipped. A
    ``CylinderVolume`` subtype with no registered display color also warns
    and falls back to a neutral gray rather than silently reusing another
    type's color.

    Args:
        stage: The target USD stage (from :func:`make_stage`).
        prim_path: Absolute USD prim path, e.g. ``"/beamline/solenoid_0"``.
        vol: A beamline volume / field object.
        transform: Internal — accumulated ``(rotation, translation)`` from
            any enclosing ``Transform`` wrappers already unwrapped by the
            caller. Leave as ``None``; recursive calls compose it themselves.
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
            name = _snake_case(_innermost_type(comp).__name__)
            add_volume(stage, f"{prim_path}/{name}_{i}", comp, transform=transform)

    elif isinstance(vol, TransformEMField):
        transform = _compose_transform(transform, vol.transform)
        add_volume(stage, prim_path, vol.field, transform=transform)

    elif isinstance(vol, TransformMaterialVolume):
        transform = _compose_transform(transform, vol.transform)
        add_volume(stage, prim_path, vol.material, transform=transform)

    elif isinstance(vol, ThinShellSolenoid):
        _add_cylinder_prim(
            stage,
            prim_path,
            vol.R,
            vol.L,
            type_colors[ThinShellSolenoid],
            _snake_case(ThinShellSolenoid.__name__),
            transform=transform,
        )

    elif isinstance(vol, ThickSolenoid):
        _add_tube_prim(
            stage,
            prim_path,
            vol.Rin,
            vol.Rout,
            vol.L,
            type_colors[ThickSolenoid],
            _snake_case(ThickSolenoid.__name__),
            transform=transform,
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
        _add_cylinder_prim(
            stage,
            prim_path,
            vol.radius,
            vol.length,
            color,
            _snake_case(type(vol).__name__),
            transform=transform,
        )

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
    trail_opacity: float = 0.35,
    animate_marker: bool = False,
    marker_color_temperature: float = 2700.0,
    marker_radius: float = 1.0,
    marker_opacity: float = 0.9,
    fps: float = 24.0,
) -> None:
    """Add particle trajectories as UsdGeom.BasisCurves polylines.

    Each trajectory is a separate ``BasisCurves`` prim under ``prim_path``.
    Positions are taken from ``states.kin.p`` (a ``Cartesian4`` array); only
    the spatial (x, y, z) components are used. The curve is drawn at reduced
    opacity (``trail_opacity``) so it reads as a "ghost trail" of the full path.

    If ``animate_marker`` is set, each trajectory also gets a glow-sphere
    sibling prim (``<prim_path>/particle_<i>_marker``, not nested under the
    curve — see below) whose position is time-sampled (one time code per
    solver sample) to travel along the path, so scrubbing/playing the
    stage's timeline shows a moving marker with the dimmed curve as its
    trail. This sets the stage's time-code range and playback rate (``fps``);
    call once per stage with the longest trajectory batch, or the range will
    be overwritten by a later call.

    The marker is an ``Xform`` group of concentric emissive/translucent
    ``UsdGeom.Sphere`` shells (see :data:`_GLOW_SHELLS`) rather than a
    ``UsdLux.SphereLight``, for two reasons: Quick Look/AR Quick Look (the
    ``.usdz`` viewer behind macOS Preview.app) excludes every ``UsdLux`` type
    from its ARKit-compatible prim allowlist and silently drops any light
    prim, and USD has no bloom/post-process schema to glow a single small
    sphere the way a real point light would — the concentric, increasingly
    transparent shells fake that "fairy light" halo instead.

    Args:
        stage: The target USD stage.
        prim_path: Base prim path, e.g. ``"/trajectories"``.
        states: A ``ParticleState`` (or batch) returned by a diffrax solver.
            Leading axis is time; an optional second axis is the particle batch.
        width: Curve display width in mm (rendered as a tube by most viewers).
        trail_opacity: Display opacity of the trail curve (0-1).
        animate_marker: If True, add a moving glow-sphere marker per trajectory.
        marker_color_temperature: Blackbody color temperature (Kelvin) of the
            marker.
        marker_radius: Radius (mm) of the marker's core shell (halo shells
            scale up from this — see :data:`_GLOW_SHELLS`).
        marker_opacity: Opacity of the marker's core shell (0-1); halo shells
            scale down from this. The core is < 1 so it stays visible as it
            passes through solid volumes.
        fps: Time codes per second for the marker animation.
    """
    _require_pxr()

    # coords shape: (n_times, 4) or (n_times, n_particles, 4)
    positions = np.asarray(states.kin.p.coords)[..., :3]  # (..., 3)

    if positions.ndim == 2:
        trajectories = [positions]
    elif positions.ndim == 3:
        trajectories = [positions[:, i, :] for i in range(positions.shape[1])]
    else:
        raise ValueError(
            f"Expected positions of shape (n_times, 3) or (n_times, n_particles, 3), "
            f"got {positions.shape}"
        )

    max_n = 0
    for i, traj in enumerate(trajectories):
        particle_path = f"{prim_path}/particle_{i}"
        _add_curve_prim(
            stage, particle_path, traj, width, _COLOR_TRAJECTORY, trail_opacity
        )
        if animate_marker:
            # A sibling of particle_path, not a child: ARKit-compliant usdz
            # (and so Quick Look/AR Quick Look) disallows nesting one Gprim
            # (Sphere) inside another (BasisCurves).
            _add_moving_glow_sphere_prim(
                stage,
                f"{prim_path}/particle_{i}_marker",
                traj,
                marker_color_temperature,
                marker_radius,
                marker_opacity,
            )
            max_n = max(max_n, traj.shape[0])

    if animate_marker and max_n > 0:
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(max_n - 1)
