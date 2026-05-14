"""
Direct AST → DXF builder.
Converts a validated JSON AST to a 2D architectural floor plan DXF
without going through LLM code generation.
"""

import math
import logging
from pathlib import Path
from typing import List, Optional
import ezdxf
from ezdxf.enums import TextEntityAlignment

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _perp(dx: float, dy: float, half_w: float):
    """Unit perpendicular scaled to half_w."""
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return 0.0, 0.0
    return -dy / length * half_w, dx / length * half_w


def _wall_pts(start, end, width: float):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    px, py = _perp(dx, dy, width / 2)
    return [
        (start[0] - px, start[1] - py),
        (end[0] - px,   end[1] - py),
        (end[0] + px,   end[1] + py),
        (start[0] + px, start[1] + py),
    ]


def _add_layers(doc):
    specs = [
        ("GRID",        8),
        ("WALLS",       1),
        ("COLUMNS",     1),
        ("BEAMS",       5),
        ("SLABS",       3),
        ("OPENINGS",    4),
        ("STAIRS",      6),
        ("ELEVATOR",    6),
        ("MEP",         2),
        ("BALCONIES",   3),
        ("HATCH_WALLS", 8),
        ("HATCH_COLS",  254),
        ("ROOMS",       7),
        ("ANNOTATIONS", 7),
        ("DIMS",        7),
        ("TITLE_BLOCK", 7),
    ]
    for name, color in specs:
        doc.layers.add(name=name, color=color)


# ── Grid ─────────────────────────────────────────────────────────────────────

def _draw_grid(msp, building: dict, site: dict, origin=(0, 0)):
    ox, oy = origin
    w = site.get("width", 20000)
    l = site.get("length", 30000)
    gx = building.get("column_grid_x", 6000)
    gy = building.get("column_grid_y", 6000)
    margin = 2000

    x = ox
    col_idx = 0
    while x <= ox + w + 1:
        label = chr(ord("A") + col_idx)
        msp.add_line((x, oy - margin), (x, oy + l + margin), dxfattribs={"layer": "GRID"})
        for y_pos in [oy - margin - 500, oy + l + margin + 500]:
            mt = msp.add_mtext(label, dxfattribs={"layer": "GRID", "char_height": 300, "insert": (x, y_pos)})
            mt.set_location((x, y_pos), attachment_point=5)
        x += gx
        col_idx += 1

    y = oy
    row_idx = 1
    while y <= oy + l + 1:
        label = str(row_idx)
        msp.add_line((ox - margin, y), (ox + w + margin, y), dxfattribs={"layer": "GRID"})
        for x_pos in [ox - margin - 500, ox + w + margin + 500]:
            mt = msp.add_mtext(label, dxfattribs={"layer": "GRID", "char_height": 300, "insert": (x_pos, y)})
            mt.set_location((x_pos, y), attachment_point=5)
        y += gy
        row_idx += 1


# ── Walls ─────────────────────────────────────────────────────────────────────

def _draw_walls(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "wall" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 200)
        pts = _wall_pts(s, e, w)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "WALLS"})
        hatch = msp.add_hatch(dxfattribs={"layer": "HATCH_WALLS"})
        hatch.set_pattern_fill("ANSI31", scale=30)
        hatch.paths.add_polyline_path(pts, is_closed=True)


# ── Columns ───────────────────────────────────────────────────────────────────

def _draw_columns(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "column" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        cx, cy = g["start"][0], g["start"][1]
        hw = g.get("width", 400) / 2
        pts = [
            (cx - hw, cy - hw), (cx + hw, cy - hw),
            (cx + hw, cy + hw), (cx - hw, cy + hw),
        ]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "COLUMNS"})
        hatch = msp.add_hatch(dxfattribs={"layer": "HATCH_COLS"})
        hatch.set_pattern_fill("SOLID")
        hatch.paths.add_polyline_path(pts, is_closed=True)


# ── Beams ─────────────────────────────────────────────────────────────────────

def _draw_beams(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "beam" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        px, py = _perp(dx, dy, g.get("width", 300) / 2)
        msp.add_line((s[0] - px, s[1] - py), (e[0] - px, e[1] - py), dxfattribs={"layer": "BEAMS"})
        msp.add_line((s[0] + px, s[1] + py), (e[0] + px, e[1] + py), dxfattribs={"layer": "BEAMS"})


# ── Slabs ─────────────────────────────────────────────────────────────────────

def _draw_slabs(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "slab" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        pts = [(s[0], s[1]), (e[0], s[1]), (e[0], e[1]), (s[0], e[1])]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SLABS"})


# ── Openings ──────────────────────────────────────────────────────────────────

def _draw_openings(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "opening" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        props = el.get("properties", {})
        subtype = props.get("subtype", "window")

        cx = (s[0] + e[0]) / 2
        cy = (s[1] + e[1]) / 2
        horiz = abs(s[0] - e[0]) >= abs(s[1] - e[1])
        w = props.get("clear_width_mm") or props.get("width") or abs(e[0] - s[0]) or 1200

        if subtype == "window":
            off = 80
            if horiz:
                for dy_off in [-off, 0, off]:
                    msp.add_line((cx - w / 2, cy + dy_off), (cx + w / 2, cy + dy_off),
                                 dxfattribs={"layer": "OPENINGS"})
            else:
                for dx_off in [-off, 0, off]:
                    msp.add_line((cx + dx_off, cy - w / 2), (cx + dx_off, cy + w / 2),
                                 dxfattribs={"layer": "OPENINGS"})
        else:  # door
            radius = w / 2
            if horiz:
                msp.add_arc((cx, cy + radius), radius, 180, 270, dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx, cy), (cx, cy + radius), dxfattribs={"layer": "OPENINGS"})
            else:
                msp.add_arc((cx + radius, cy), radius, 90, 180, dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx, cy), (cx + radius, cy), dxfattribs={"layer": "OPENINGS"})


# ── Staircases ────────────────────────────────────────────────────────────────

def _draw_stairs(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "staircase" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 1200)
        pts = [(s[0], s[1]), (s[0] + w, s[1]), (s[0] + w, e[1]), (s[0], e[1])]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "STAIRS"})
        length = abs(e[1] - s[1])
        step = 250
        y = s[1]
        while y < e[1]:
            msp.add_line((s[0], y), (s[0] + w, y), dxfattribs={"layer": "STAIRS"})
            y += step
        cx = s[0] + w / 2
        mid_y = s[1] + length / 2
        msp.add_line((cx, mid_y - 400), (cx, mid_y + 400), dxfattribs={"layer": "STAIRS"})
        msp.add_line((cx, mid_y + 400), (cx - 150, mid_y + 200), dxfattribs={"layer": "STAIRS"})
        msp.add_line((cx, mid_y + 400), (cx + 150, mid_y + 200), dxfattribs={"layer": "STAIRS"})
        mt = msp.add_mtext("UP", dxfattribs={"layer": "STAIRS", "char_height": 180, "insert": (cx, mid_y - 700)})
        mt.set_location((cx, mid_y - 700), attachment_point=5)


# ── Elevators ─────────────────────────────────────────────────────────────────

def _draw_elevators(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "elevator" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        w = g.get("width", 1600)
        d = abs(e[1] - s[1]) or 2100
        pts = [(s[0], s[1]), (s[0] + w, s[1]), (s[0] + w, s[1] + d), (s[0], s[1] + d)]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "ELEVATOR"})
        cx, cy = s[0] + w / 2, s[1] + d / 2
        half = min(w, d) * 0.35
        msp.add_line((cx - half, cy - half), (cx + half, cy + half), dxfattribs={"layer": "ELEVATOR"})
        msp.add_line((cx + half, cy - half), (cx - half, cy + half), dxfattribs={"layer": "ELEVATOR"})
        mt = msp.add_mtext("LIFT", dxfattribs={"layer": "ELEVATOR", "char_height": 180, "insert": (cx, cy + half + 200)})
        mt.set_location((cx, cy + half + 200), attachment_point=5)


# ── MEP shafts ────────────────────────────────────────────────────────────────

def _draw_mep(msp, elements: List[dict], mep_data: dict, floor: int):
    for el in elements:
        if el.get("type") != "mep_shaft" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s = g["start"]
        w = g.get("width", 600)
        d = g.get("height", 600)
        pts = [(s[0], s[1]), (s[0] + w, s[1]), (s[0] + w, s[1] + d), (s[0], s[1] + d)]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "MEP"})
        services = el.get("properties", {}).get("services", ["MEP"])
        label = services[0][:8] if services else "MEP"
        cx, cy = s[0] + w / 2, s[1] + d / 2
        mt = msp.add_mtext(label, dxfattribs={"layer": "MEP", "char_height": 120, "insert": (cx, cy)})
        mt.set_location((cx, cy), attachment_point=5)

    # Draw MEP panels from mep_data
    elec = mep_data.get("electrical", {})
    main = elec.get("main_panel") if isinstance(elec.get("main_panel"), (list, tuple)) else None
    if main and len(main) >= 2 and (len(main) < 3 or main[2] == floor):
        px, py = main[0], main[1]
        pts = [(px - 300, py - 300), (px + 300, py - 300), (px + 300, py + 300), (px - 300, py + 300)]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "MEP"})
        mt = msp.add_mtext("E-PANEL", dxfattribs={"layer": "MEP", "char_height": 120, "insert": (px, py)})
        mt.set_location((px, py), attachment_point=5)


# ── Balconies ─────────────────────────────────────────────────────────────────

def _draw_balconies(msp, elements: List[dict], floor: int):
    for el in elements:
        if el.get("type") != "balcony" or el.get("floor", 0) != floor:
            continue
        g = el["geometry"]
        s, e = g["start"], g["end"]
        pts = [(s[0], s[1]), (e[0], s[1]), (e[0], e[1]), (s[0], e[1])]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "BALCONIES"})


# ── Rooms ─────────────────────────────────────────────────────────────────────

def _draw_rooms(msp, floors: List[dict], floor_level: int):
    for fl in floors:
        if fl.get("level", 0) != floor_level:
            continue
        for room in fl.get("rooms", []):
            perim = room.get("perimeter")
            if not perim or len(perim) < 3:
                continue
            cx = sum(p[0] for p in perim) / len(perim)
            cy = sum(p[1] for p in perim) / len(perim)
            area = room.get("area", room.get("net_area", 0))
            name = room.get("name", room.get("type", "Room"))
            label = f"{name}\n{area:.1f} m2"
            mt = msp.add_mtext(label, dxfattribs={"layer": "ROOMS", "char_height": 180, "insert": (cx, cy)})
            mt.set_location((cx, cy), attachment_point=5)


# ── Dimensions ────────────────────────────────────────────────────────────────

def _draw_dimensions(msp, site: dict, origin=(0, 0)):
    ox, oy = origin
    w = site.get("width", 20000)
    l = site.get("length", 30000)
    off = 2000

    dim_w = msp.add_aligned_dim(
        p1=(ox, oy), p2=(ox + w, oy), distance=off,
        dimstyle="EZDXF", override={"dimtxt": 250}, dxfattribs={"layer": "DIMS"}
    )
    dim_w.render()

    dim_l = msp.add_aligned_dim(
        p1=(ox, oy), p2=(ox, oy + l), distance=off,
        dimstyle="EZDXF", override={"dimtxt": 250}, dxfattribs={"layer": "DIMS"}
    )
    dim_l.render()


# ── Title block ───────────────────────────────────────────────────────────────

def _draw_title_block(msp, building: dict, site: dict, floor_label: str, origin=(0, 0)):
    ox, oy = origin
    w = site.get("width", 20000)
    tb_x = ox + w + 4000
    tb_y = oy + 1000
    tb_w, tb_h = 8000, 4000

    pts = [(tb_x, tb_y), (tb_x + tb_w, tb_y), (tb_x + tb_w, tb_y + tb_h), (tb_x, tb_y + tb_h)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "TITLE_BLOCK"})

    lines = [
        (f"{building.get('type','').upper()} BUILDING", 300, tb_h - 400),
        (floor_label, 250, tb_h - 800),
        (f"Total area: {building.get('total_area', 0):.0f} m2", 200, tb_h - 1200),
        (f"Floors: {building.get('floors', 0)}", 200, tb_h - 1500),
        (f"Structure: {building.get('structural_system', '')}", 200, tb_h - 1800),
        ("SCALE 1:100", 180, tb_h - 2200),
        ("DRAWING NO. A-001", 180, tb_h - 2500),
    ]
    for text, height, dy in lines:
        mt = msp.add_mtext(text, dxfattribs={
            "layer": "TITLE_BLOCK", "char_height": height,
            "insert": (tb_x + tb_w / 2, tb_y + dy)
        })
        mt.set_location((tb_x + tb_w / 2, tb_y + dy), attachment_point=5)

    # North arrow
    ax, ay = tb_x + 700, tb_y + 500
    msp.add_lwpolyline([(ax, ay + 400), (ax - 180, ay), (ax + 180, ay)], close=True,
                       dxfattribs={"layer": "TITLE_BLOCK"})
    mt = msp.add_mtext("N", dxfattribs={"layer": "TITLE_BLOCK", "char_height": 200, "insert": (ax, ay + 600)})
    mt.set_location((ax, ay + 600), attachment_point=5)


# ── Floor plan ────────────────────────────────────────────────────────────────

def _draw_floor_plan(msp, ast: dict, floor: int, origin=(0, 0)):
    elements = ast.get("elements", [])
    floors = ast.get("floors", [])
    mep = ast.get("mep", {})
    site = ast.get("site", {})
    building = ast.get("building", {})

    _draw_grid(msp, building, site, origin)
    _draw_slabs(msp, elements, floor)
    _draw_walls(msp, elements, floor)
    _draw_columns(msp, elements, floor)
    _draw_beams(msp, elements, floor)
    _draw_openings(msp, elements, floor)
    _draw_stairs(msp, elements, floor)
    _draw_elevators(msp, elements, floor)
    _draw_balconies(msp, elements, floor)
    _draw_mep(msp, elements, mep, floor)
    _draw_rooms(msp, floors, floor)
    _draw_dimensions(msp, site, origin)

    fl_info = next((f for f in floors if f.get("level") == floor), {})
    label = f"FLOOR {floor} — ELEVATION {fl_info.get('elevation', 0):.0f}mm"
    _draw_title_block(msp, building, site, label, origin)


# ── Public API ────────────────────────────────────────────────────────────────

def build_dxf(ast: dict, output_path: str = "outputs/building_2d.dxf") -> str:
    """
    Convert AST to a 2D DXF floor plan.
    Draws the ground floor plan (level 0) and the first typical floor (level 1) side by side.
    Returns the saved file path.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    _add_layers(doc)

    site = ast.get("site", {})
    building = ast.get("building", {})
    floors = ast.get("floors", [])
    floor_levels = sorted({f.get("level", 0) for f in floors})

    site_w = site.get("width", 20000)
    gap = 8000  # horizontal gap between floor plans

    drawn_floors = floor_levels[:2]  # ground + first typical
    for i, lvl in enumerate(drawn_floors):
        ox = i * (site_w + gap)
        _draw_floor_plan(msp, ast, lvl, origin=(ox, 0))

    # Viewport: center on all drawn plans
    total_w = len(drawn_floors) * (site_w + gap)
    site_l = site.get("length", 30000)
    cx = total_w / 2
    cy = site_l / 2
    doc.set_modelspace_vport(height=site_l + 16000, center=(cx, cy))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)
    logger.info(f"DXF saved: {output_path}")
    return output_path
