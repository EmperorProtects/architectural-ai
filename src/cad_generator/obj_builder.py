"""
Direct AST → OBJ 3D builder.
Converts a validated JSON AST to a Wavefront OBJ + MTL file for 3D viewing.
No LLM required — pure algorithmic geometry.
"""

import math
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Material → OBJ mtl name mapping
MATERIALS = {
    "wall_exterior":    ("wall_ext",   (0.85, 0.80, 0.72)),  # warm grey
    "wall_interior":    ("wall_int",   (0.95, 0.92, 0.88)),  # off-white
    "column":          ("column",     (0.60, 0.60, 0.62)),  # concrete grey
    "beam":            ("beam",       (0.55, 0.55, 0.58)),  # slightly darker
    "slab_top":        ("slab",       (0.70, 0.68, 0.65)),  # floor slab
    "window":          ("window",     (0.50, 0.72, 0.90)),  # glass blue, semi
    "door":            ("door",       (0.55, 0.38, 0.25)),  # wood brown
    "stair":           ("stair",      (0.75, 0.72, 0.68)),  # light concrete
    "elevator":        ("elevator",   (0.45, 0.45, 0.50)),  # dark metal
    "mep_shaft":       ("mep",        (0.80, 0.50, 0.20)),  # orange
    "balcony":         ("balcony",    (0.80, 0.80, 0.78)),  # light grey
}


def _write_mtl(path: str):
    lines = ["# Architectural AI material library\n"]
    for key, (name, (r, g, b)) in MATERIALS.items():
        lines += [
            f"newmtl {name}",
            f"Kd {r:.3f} {g:.3f} {b:.3f}",
            "Ka 0.1 0.1 0.1",
            "Ks 0.05 0.05 0.05",
            "Ns 10",
            "d 1.0" if "window" not in key else "d 0.35",
            "",
        ]
    Path(path).write_text("\n".join(lines))


class _OBJWriter:
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, ...]] = []
        self.groups: List[Tuple[str, str, int, int]] = []  # (group, mtl, face_start, face_end)
        self._current_group = ""
        self._current_mtl = ""
        self._face_start = 0

    def _v(self, x, y, z) -> int:
        self.vertices.append((x / 1000.0, y / 1000.0, z / 1000.0))  # mm → m
        return len(self.vertices)  # 1-indexed

    def _box(self, x0, y0, z0, x1, y1, z1):
        """Add an axis-aligned box with 6 quads."""
        v = [
            self._v(x0, y0, z0), self._v(x1, y0, z0),
            self._v(x1, y1, z0), self._v(x0, y1, z0),
            self._v(x0, y0, z1), self._v(x1, y0, z1),
            self._v(x1, y1, z1), self._v(x0, y1, z1),
        ]
        a, b, c, d, e, f, g, h = v
        self.faces += [
            (d, c, b, a),  # bottom
            (e, f, g, h),  # top
            (a, b, f, e),  # front
            (b, c, g, f),  # right
            (c, d, h, g),  # back
            (d, a, e, h),  # left
        ]

    def _extrude_wall(self, x0, y0, x1, y1, z_bot, z_top, thickness):
        """Extrude a wall segment (line → thick box in 3D)."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return
        px = -dy / length * thickness / 2
        py = dx / length * thickness / 2

        p = [
            (x0 - px, y0 - py), (x1 - px, y1 - py),
            (x1 + px, y1 + py), (x0 + px, y0 + py),
        ]
        vb = [self._v(pt[0], pt[1], z_bot) for pt in p]
        vt = [self._v(pt[0], pt[1], z_top) for pt in p]
        a, b, c, d = vb
        e, f, g, h = vt
        self.faces += [
            (d, c, b, a),      # bottom
            (e, f, g, h),      # top
            (a, b, f, e),      # side 1
            (b, c, g, f),      # side 2
            (c, d, h, g),      # side 3
            (d, a, e, h),      # side 4
        ]

    def begin_group(self, name: str, mtl_key: str):
        self._current_group = name
        self._current_mtl = MATERIALS.get(mtl_key, ("default", (0.7, 0.7, 0.7)))[0]
        self._face_start = len(self.faces)

    def end_group(self):
        if len(self.faces) > self._face_start:
            self.groups.append((
                self._current_group,
                self._current_mtl,
                self._face_start,
                len(self.faces),
            ))

    def write(self, obj_path: str, mtl_filename: str):
        lines = [
            f"# Architectural AI — 3D model",
            f"mtllib {mtl_filename}",
            "",
        ]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        lines.append("")

        g_ptr = 0
        groups = sorted(self.groups, key=lambda x: x[2])
        face_idx = 0
        for gi, (grp, mtl, fs, fe) in enumerate(groups):
            lines.append(f"g {grp}")
            lines.append(f"usemtl {mtl}")
            for face in self.faces[fs:fe]:
                lines.append("f " + " ".join(str(v) for v in face))
        Path(obj_path).write_text("\n".join(lines))


# ── Element builders ──────────────────────────────────────────────────────────

def _add_walls(writer: _OBJWriter, elements: list):
    writer.begin_group("walls", "wall_exterior")
    for el in elements:
        if el.get("type") != "wall":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        z_bot = s[2]
        z_top = z_bot + g.get("height", 3000)
        props = el.get("properties", {})
        mtl = "wall_exterior" if props.get("subtype") == "exterior" else "wall_interior"
        writer.begin_group(f"wall_{el['id']}", mtl)
        writer._extrude_wall(s[0], s[1], e[0], e[1], z_bot, z_top, g.get("width", 200))
        writer.end_group()
    writer.end_group()


def _add_columns(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "column":
            continue
        g = el["geometry"]
        cx, cy = g["start"][0], g["start"][1]
        hw = g.get("width", 400) / 2
        z_bot = g["start"][2]
        z_top = g["end"][2] if "end" in g and g["end"] else z_bot + g.get("height", 3000)
        writer.begin_group(f"col_{el['id']}", "column")
        writer._box(cx - hw, cy - hw, z_bot, cx + hw, cy + hw, z_top)
        writer.end_group()


def _add_beams(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "beam":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 300)
        h = g.get("height", 600)
        z_top = s[2]
        z_bot = z_top - h
        writer.begin_group(f"beam_{el['id']}", "beam")
        writer._extrude_wall(s[0], s[1], e[0], e[1], z_bot, z_top, w)
        writer.end_group()


def _add_slabs(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "slab":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        z = s[2]
        thickness = el.get("properties", {}).get("structural_thickness_mm", 200)
        writer.begin_group(f"slab_{el['id']}", "slab_top")
        writer._box(min(s[0], e[0]), min(s[1], e[1]), z - thickness,
                    max(s[0], e[0]), max(s[1], e[1]), z)
        writer.end_group()


def _add_stairs(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "staircase":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 1200)
        props = el.get("properties", {})
        tread = props.get("tread_depth_mm", 300)
        riser = props.get("riser_height_mm", 150)
        n_treads = int(abs(e[1] - s[1]) / tread) or 10
        writer.begin_group(f"stair_{el['id']}", "stair")
        for i in range(n_treads):
            tx = s[0]
            ty = s[1] + i * tread
            tz = s[2] + i * riser
            writer._box(tx, ty, tz, tx + w, ty + tread, tz + riser)
        writer.end_group()


def _add_elevators(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "elevator":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 1600)
        props = el.get("properties", {})
        d = props.get("shaft_depth_mm", 2100) or abs(e[1] - s[1]) or 2100
        floors = el.get("properties", {}).get("floors_served", [0])
        total_h = len(floors) * 3000 if floors else 15000
        writer.begin_group(f"elevator_{el['id']}", "elevator")
        # Shaft walls (thin skins)
        for pts, side in [
            ((s[0], s[1], s[0] + 100, s[1] + d), "front"),
            ((s[0] + w - 100, s[1], s[0] + w, s[1] + d), "back"),
            ((s[0], s[1], s[0] + w, s[1] + 100), "left"),
            ((s[0], s[1] + d - 100, s[0] + w, s[1] + d), "right"),
        ]:
            x0, y0, x1, y1 = pts
            writer._box(x0, y0, s[2], x1, y1, s[2] + total_h)
        writer.end_group()


def _add_balconies(writer: _OBJWriter, elements: list):
    for el in elements:
        if el.get("type") != "balcony":
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        props = el.get("properties", {})
        thickness = props.get("slab_thickness_mm", 150)
        railing_h = props.get("railing", {}).get("height_mm", 1100) if isinstance(props.get("railing"), dict) else 1100
        z = s[2]
        writer.begin_group(f"balcony_{el['id']}", "balcony")
        # Slab
        writer._box(min(s[0], e[0]), min(s[1], e[1]), z - thickness,
                    max(s[0], e[0]), max(s[1], e[1]), z)
        # Railing (thin shell)
        writer._box(min(s[0], e[0]), min(s[1], e[1]) - 50, z,
                    max(s[0], e[0]), min(s[1], e[1]) + 50, z + railing_h)
        writer.end_group()


# ── Public API ────────────────────────────────────────────────────────────────

def build_obj(ast: dict, output_path: str = "outputs/building_3d.obj") -> str:
    """
    Convert AST to a 3D Wavefront OBJ model.
    Returns the saved file path.
    """
    writer = _OBJWriter()
    elements = ast.get("elements", [])

    _add_walls(writer, elements)
    _add_columns(writer, elements)
    _add_beams(writer, elements)
    _add_slabs(writer, elements)
    _add_stairs(writer, elements)
    _add_elevators(writer, elements)
    _add_balconies(writer, elements)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mtl_path = str(Path(output_path).with_suffix(".mtl"))
    mtl_filename = Path(mtl_path).name

    _write_mtl(mtl_path)
    writer.write(output_path, mtl_filename)

    vcount = len(writer.vertices)
    fcount = len(writer.faces)
    logger.info(f"OBJ saved: {output_path} — {vcount} vertices, {fcount} faces")
    return output_path
