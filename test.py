import ezdxf
from ezdxf import units, zoom
from ezdxf.enums import TextEntityAlignment
import math

def _add_layers(doc):
    doc.layers.add(name="GRID", color=8)
    doc.layers.add(name="WALLS", color=1)
    doc.layers.add(name="COLUMNS", color=1)
    doc.layers.add(name="BEAMS", color=5)
    doc.layers.add(name="SLABS", color=3)
    doc.layers.add(name="OPENINGS", color=4)
    doc.layers.add(name="STAIRS", color=6)
    doc.layers.add(name="ELEVATOR", color=6)
    doc.layers.add(name="MEP", color=2)
    doc.layers.add(name="HATCH_WALLS", color=8)
    doc.layers.add(name="HATCH_COLS", color=254)
    doc.layers.add(name="ROOMS", color=7)
    doc.layers.add(name="ANNOTATIONS", color=7)
    doc.layers.add(name="DIMS", color=7)
    doc.layers.add(name="TITLE_BLOCK", color=7)

def _draw_grid(msp, building, site):
    # Building footprint: 2000,2000 to 18000,28000
    # Grid spacing 6000mm
    min_x, min_y = 2000, 2000
    max_x, max_y = 18000, 28000
    grid_x = building["column_grid_x"]
    grid_y = building["column_grid_y"]
    
    # Y grid lines (vertical) - A, B, C...
    x_lines = []
    grid_labels = []
    x = min_x
    label_idx = 0
    while x <= max_x:
        x_lines.append(x)
        grid_labels.append(chr(ord('A') + label_idx))
        x += grid_x
        label_idx += 1
    
    for i, x in enumerate(x_lines):
        msp.add_line((x, min_y - 1000), (x, max_y + 1000), dxfattribs={"layer": "GRID"})
        # Grid label at both ends
        mt_bottom = msp.add_mtext(grid_labels[i], dxfattribs={"layer": "GRID", "char_height": 300, "insert": (x, min_y - 1500)})
        mt_bottom.set_location((x, min_y - 1500), attachment_point=5)
        mt_top = msp.add_mtext(grid_labels[i], dxfattribs={"layer": "GRID", "char_height": 300, "insert": (x, max_y + 1500)})
        mt_top.set_location((x, max_y + 1500), attachment_point=5)
    
    # X grid lines (horizontal) - 1, 2, 3...
    y_lines = []
    y_labels = []
    y = min_y
    label_idx = 1
    while y <= max_y:
        y_lines.append(y)
        y_labels.append(str(label_idx))
        y += grid_y
        label_idx += 1
    
    for i, y in enumerate(y_lines):
        msp.add_line((min_x - 1000, y), (max_x + 1000, y), dxfattribs={"layer": "GRID"})
        # Grid label at both ends
        mt_left = msp.add_mtext(y_labels[i], dxfattribs={"layer": "GRID", "char_height": 300, "insert": (min_x - 1500, y)})
        mt_left.set_location((min_x - 1500, y), attachment_point=5)
        mt_right = msp.add_mtext(y_labels[i], dxfattribs={"layer": "GRID", "char_height": 300, "insert": (max_x + 1500, y)})
        mt_right.set_location((max_x + 1500, y), attachment_point=5)

def _draw_walls(msp, elements):
    walls = [e for e in elements if e["type"] == "wall" and e["floor"] == 0]  # Ground floor plan
    
    for wall in walls:
        start = wall["geometry"]["start"]
        end = wall["geometry"]["end"]
        width = wall["geometry"]["width"]
        
        # Calculate perpendicular offset for wall thickness
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            continue
        
        # Unit perpendicular vector
        perp_x = -dy / length * width / 2
        perp_y = dx / length * width / 2
        
        # Wall outline points
        pts = [
            (start[0] - perp_x, start[1] - perp_y),
            (end[0] - perp_x, end[1] - perp_y),
            (end[0] + perp_x, end[1] + perp_y),
            (start[0] + perp_x, start[1] + perp_y)
        ]
        
        # Draw wall polyline
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "WALLS"})
        
        # Add hatch
        hatch = msp.add_hatch(dxfattribs={"layer": "HATCH_WALLS"})
        hatch.set_pattern_fill("ANSI31", scale=30)
        hatch.paths.add_polyline_path(pts, is_closed=True)

def _draw_columns(msp, elements):
    columns = [e for e in elements if e["type"] == "column"]
    
    for col in columns:
        cx, cy = col["geometry"]["start"][0], col["geometry"]["start"][1]
        w = col["geometry"]["width"]
        hw = w / 2
        
        # Column square
        pts = [
            (cx - hw, cy - hw),
            (cx + hw, cy - hw),
            (cx + hw, cy + hw),
            (cx - hw, cy + hw)
        ]
        
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "COLUMNS"})
        
        # Add solid hatch
        hatch = msp.add_hatch(dxfattribs={"layer": "HATCH_COLS"})
        hatch.set_pattern_fill("SOLID")
        hatch.paths.add_polyline_path(pts, is_closed=True)

def _draw_beams(msp, elements):
    beams = [e for e in elements if e["type"] == "beam" and e["floor"] == 0]  # Ground floor plan
    
    for beam in beams:
        start = beam["geometry"]["start"]
        end = beam["geometry"]["end"]
        width = beam["geometry"]["width"]
        
        # Calculate perpendicular offset for beam width
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            continue
        
        # Unit perpendicular vector
        perp_x = -dy / length * width / 2
        perp_y = dx / length * width / 2
        
        # Draw double lines
        msp.add_line((start[0] - perp_x, start[1] - perp_y), (end[0] - perp_x, end[1] - perp_y), 
                     dxfattribs={"layer": "BEAMS"})
        msp.add_line((start[0] + perp_x, start[1] + perp_y), (end[0] + perp_x, end[1] + perp_y), 
                     dxfattribs={"layer": "BEAMS"})

def _draw_slabs(msp, elements):
    slabs = [e for e in elements if e["type"] == "slab" and e["floor"] == 0]  # Ground floor plan
    
    for slab in slabs:
        start = slab["geometry"]["start"]
        end = slab["geometry"]["end"]
        
        # Slab boundary rectangle
        pts = [
            (start[0], start[1]),
            (end[0], start[1]),
            (end[0], end[1]),
            (start[0], end[1])
        ]
        
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SLABS"})
        
        # Cross-hatch pattern
        step = 1000
        for x in range(int(start[0]), int(end[0]), step):
            msp.add_line((x, start[1]), (x, end[1]), dxfattribs={"layer": "SLABS"})
        for y in range(int(start[1]), int(end[1]), step):
            msp.add_line((start[0], y), (end[0], y), dxfattribs={"layer": "SLABS"})

def _draw_openings(msp, elements):
    openings = [e for e in elements if e["type"] == "opening" and e["floor"] == 0]
    
    for opening in openings:
        start = opening["geometry"]["start"]
        end = opening["geometry"]["end"]
        subtype = opening["properties"]["subtype"]
        width = opening["properties"]["width"]
        
        if subtype == "window":
            # Three parallel lines for glazing
            cx = (start[0] + end[0]) / 2
            cy = (start[1] + end[1]) / 2
            
            # Determine orientation
            if abs(start[0] - end[0]) > abs(start[1] - end[1]):  # Horizontal
                offset = 100
                msp.add_line((start[0], cy - offset), (end[0], cy - offset), dxfattribs={"layer": "OPENINGS"})
                msp.add_line((start[0], cy), (end[0], cy), dxfattribs={"layer": "OPENINGS"})
                msp.add_line((start[0], cy + offset), (end[0], cy + offset), dxfattribs={"layer": "OPENINGS"})
            else:  # Vertical
                offset = 100
                msp.add_line((cx - offset, start[1]), (cx - offset, end[1]), dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx, start[1]), (cx, end[1]), dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx + offset, start[1]), (cx + offset, end[1]), dxfattribs={"layer": "OPENINGS"})
        
        elif subtype == "door":
            # Arc sweep + door leaf line
            cx = (start[0] + end[0]) / 2
            cy = (start[1] + end[1]) / 2
            
            # Door arc (90 degree sweep)
            radius = width / 2
            if abs(start[0] - end[0]) > abs(start[1] - end[1]):  # Horizontal door
                arc_center = (cx, cy + radius)
                msp.add_arc(arc_center, radius, 180, 270, dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx, cy), (cx, cy + radius), dxfattribs={"layer": "OPENINGS"})
            else:  # Vertical door
                arc_center = (cx + radius, cy)
                msp.add_arc(arc_center, radius, 90, 180, dxfattribs={"layer": "OPENINGS"})
                msp.add_line((cx, cy), (cx + radius, cy), dxfattribs={"layer": "OPENINGS"})

def _draw_stairs_elevators(msp, elements):
    stairs = [e for e in elements if e["type"] == "staircase"]
    
    for stair in stairs:
        start = stair["geometry"]["start"]
        end = stair["geometry"]["end"]
        width = stair["geometry"]["width"]
        
        # Staircase rectangle
        pts = [
            (start[0], start[1]),
            (start[0] + width, start[1]),
            (start[0] + width, end[1]),
            (start[0], end[1])
        ]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "STAIRS"})
        
        # Flight lines and treads
        flight_length = end[1] - start[1]
        tread_step = 250
        for y in range(int(start[1]), int(end[1]), tread_step):
            msp.add_line((start[0], y), (start[0] + width, y), dxfattribs={"layer": "STAIRS"})
        
        # Direction arrow and UP label
        cx = start[0] + width / 2
        cy = start[1] + flight_length / 2
        msp.add_line((cx, cy - 500), (cx, cy + 500), dxfattribs={"layer": "STAIRS"})
        msp.add_line((cx, cy + 500), (cx - 200, cy + 300), dxfattribs={"layer": "STAIRS"})
        msp.add_line((cx, cy + 500), (cx + 200, cy + 300), dxfattribs={"layer": "STAIRS"})
        
        mt_up = msp.add_mtext("UP", dxfattribs={"layer": "STAIRS", "char_height": 200, "insert": (cx, cy - 800)})
        mt_up.set_location((cx, cy - 800), attachment_point=5)

def _draw_mep(msp, mep):
    # MEP shaft as labeled rectangles
    electrical = mep["electrical"]
    plumbing = mep["plumbing"]
    
    # Electrical panel
    main_panel = electrical["main_panel"]
    pts = [
        (main_panel[0] - 500, main_panel[1] - 500),
        (main_panel[0] + 500, main_panel[1] - 500),
        (main_panel[0] + 500, main_panel[1] + 500),
        (main_panel[0] - 500, main_panel[1] + 500)
    ]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "MEP"})
    mt_elec = msp.add_mtext("ELECTRICAL\nPANEL", dxfattribs={"layer": "MEP", "char_height": 150, "insert": (main_panel[0], main_panel[1])})
    mt_elec.set_location((main_panel[0], main_panel[1]), attachment_point=5)
    
    # Wet risers
    for riser in plumbing["wet_risers"]:
        pts = [
            (riser[0] - 500, riser[1] - 500),
            (riser[0] + 500, riser[1] - 500),
            (riser[0] + 500, riser[1] + 500),
            (riser[0] - 500, riser[1] + 500)
        ]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "MEP"})
        mt_wet = msp.add_mtext("WET\nSHAFT", dxfattribs={"layer": "MEP", "char_height": 150, "insert": (riser[0], riser[1])})
        mt_wet.set_location((riser[0], riser[1]), attachment_point=5)

def _draw_rooms(msp, floors):
    floor_0 = floors[0]  # Ground floor
    
    for room in floor_0["rooms"]:
        perimeter = room["perimeter"]
        area = room["area"]
        name = room["name"]
        
        # Calculate room center for label
        cx = sum(p[0] for p in perimeter) / len(perimeter)
        cy = sum(p[1] for p in perimeter) / len(perimeter)
        
        # Room label
        label_text = f"{name}\n{area:.1f} m²"
        mt_room = msp.add_mtext(label_text, dxfattribs={"layer": "ROOMS", "char_height": 200, "insert": (cx, cy)})
        mt_room.set_location((cx, cy), attachment_point=5)

def _draw_dimensions(msp, building, site):
    # Overall building dimensions (offset 1500mm outside)
    min_x, min_y = 2000, 2000
    max_x, max_y = 18000, 28000
    
    # Building width dimension (bottom)
    dim_w = msp.add_aligned_dim(p1=(min_x, min_y), p2=(max_x, min_y), distance=1500,
                               dimstyle="EZDXF", override={"dimtxt": 250}, 
                               dxfattribs={"layer": "DIMS"})
    dim_w.render()
    
    # Building depth dimension (left)
    dim_d = msp.add_aligned_dim(p1=(min_x, min_y), p2=(min_x, max_y), distance=1500,
                               dimstyle="EZDXF", override={"dimtxt": 250}, 
                               dxfattribs={"layer": "DIMS"})
    dim_d.render()
    
    # Column grid spacing
    grid_x = building["column_grid_x"]
    
    # Grid spacing along X (top)
    dim_gx1 = msp.add_aligned_dim(p1=(2000, max_y), p2=(8000, max_y), distance=800,
                                 dimstyle="EZDXF", override={"dimtxt": 250}, 
                                 dxfattribs={"layer": "DIMS"})
    dim_gx1.render()
    
    dim_gx2 = msp.add_aligned_dim(p1=(8000, max_y), p2=(14000, max_y), distance=800,
                                 dimstyle="EZDXF", override={"dimtxt": 250}, 
                                 dxfattribs={"layer": "DIMS"})
    dim_gx2.render()
    
    # Grid spacing along Y (right)
    dim_gy1 = msp.add_aligned_dim(p1=(max_x, 2000), p2=(max_x, 8000), distance=800,
                                 dimstyle="EZDXF", override={"dimtxt": 250}, 
                                 dxfattribs={"layer": "DIMS"})
    dim_gy1.render()
    
    dim_gy2 = msp.add_aligned_dim(p1=(max_x, 8000), p2=(max_x, 14000), distance=800,
                                 dimstyle="EZDXF", override={"dimtxt": 250}, 
                                 dxfattribs={"layer": "DIMS"})
    dim_gy2.render()

def _draw_title_block(msp, building):
    # Title block position (bottom-right, outside footprint)
    tb_x = 22000
    tb_y = 1000
    tb_w = 4000
    tb_h = 2000
    
    # Title block rectangle
    pts = [
        (tb_x, tb_y),
        (tb_x + tb_w, tb_y),
        (tb_x + tb_w, tb_y + tb_h),
        (tb_x, tb_y + tb_h)
    ]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "TITLE_BLOCK"})
    
    # Project information
    mt_proj = msp.add_mtext("5-STORY RESIDENTIAL BUILDING", 
                           dxfattribs={"layer": "TITLE_BLOCK", "char_height": 200, 
                                     "insert": (tb_x + tb_w/2, tb_y + tb_h - 400)})
    mt_proj.set_location((tb_x + tb_w/2, tb_y + tb_h - 400), attachment_point=5)
    
    mt_scale = msp.add_mtext("SCALE 1:100", 
                            dxfattribs={"layer": "TITLE_BLOCK", "char_height": 150, 
                                      "insert": (tb_x + tb_w/2, tb_y + tb_h - 800)})
    mt_scale.set_location((tb_x + tb_w/2, tb_y + tb_h - 800), attachment_point=5)
    
    mt_dwg = msp.add_mtext("DRAWING NO. A-001", 
                          dxfattribs={"layer": "TITLE_BLOCK", "char_height": 150, 
                                    "insert": (tb_x + tb_w/2, tb_y + tb_h - 1200)})
    mt_dwg.set_location((tb_x + tb_w/2, tb_y + tb_h - 1200), attachment_point=5)
    
    # North arrow (triangle)
    arrow_x = tb_x + 500
    arrow_y = tb_y + 500
    arrow_pts = [
        (arrow_x, arrow_y + 300),
        (arrow_x - 150, arrow_y),
        (arrow_x + 150, arrow_y)
    ]
    msp.add_lwpolyline(arrow_pts, close=True, dxfattribs={"layer": "TITLE_BLOCK"})
    
    mt_north = msp.add_mtext("N", dxfattribs={"layer": "TITLE_BLOCK", "char_height": 150, 
                                            "insert": (arrow_x, arrow_y - 300)})
    mt_north.set_location((arrow_x, arrow_y - 300), attachment_point=5)

def generate_drawing():
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # millimeters
    msp = doc.modelspace()
    
    # Building and site data from AST
    building = {
        "type": "residential",
        "floors": 5,
        "column_grid_x": 6000.0,
        "column_grid_y": 6000.0
    }
    
    site = {
        "width": 20000.0,
        "length": 30000.0
    }
    
    # Elements from AST (subset for ground floor)
    elements = [
        {"id": "wall_ext_f0_north", "type": "wall", "floor": 0, 
         "geometry": {"start": [2000, 28000, 0], "end": [18000, 28000, 0], "width": 400.0}},
        {"id": "wall_ext_f0_south", "type": "wall", "floor": 0, 
         "geometry": {"start": [2000, 2000, 0], "end": [18000, 2000, 0], "width": 400.0}},
        {"id": "wall_ext_f0_east", "type": "wall", "floor": 0, 
         "geometry": {"start": [18000, 2000, 0], "end": [18000, 28000, 0], "width": 400.0}},
        {"id": "wall_ext_f0_west", "type": "wall", "floor": 0, 
         "geometry": {"start": [2000, 2000, 0], "end": [2000, 28000, 0], "width": 400.0}},
        {"id": "wall_int_f0_001", "type": "wall", "floor": 0, 
         "geometry": {"start": [8000, 2400, 0], "end": [8000, 27600, 0], "width": 200.0}},
        {"id": "wall_int_f0_002", "type": "wall", "floor": 0, 
         "geometry": {"start": [12000, 2400, 0], "end": [12000, 27600, 0], "width": 200.0}},
        {"id": "col_001", "type": "column", "floor": 0, 
         "geometry": {"start": [8000, 8000, 0], "width": 400.0}},
        {"id": "col_002", "type": "column", "floor": 0, 
         "geometry": {"start": [14000, 8000, 0], "width": 400.0}},
        {"id": "col_003", "type": "column", "floor": 0, 
         "geometry": {"start": [8000, 14000, 0], "width": 400.0}},
        {"id": "col_004", "type": "column", "floor": 0, 
         "geometry": {"start": [14000, 14000, 0], "width": 400.0}},
        {"id": "col_005", "type": "column", "floor": 0, 
         "geometry": {"start": [8000, 20000, 0], "width": 400.0}},
        {"id": "col_006", "type": "column", "floor": 0, 
         "geometry": {"start": [14000, 20000, 0], "width": 400.0}},
        {"id": "beam_f0_001", "type": "beam", "floor": 0, 
         "geometry": {"start": [8000, 8000, 3000], "end": [14000, 8000, 3000], "width": 300.0}},
        {"id": "beam_f0_002", "type": "beam", "floor": 0, 
         "geometry": {"start": [8000, 14000, 3000], "end": [14000, 14000, 3000], "width": 300.0}},
        {"id": "beam_f0_003", "type": "beam", "floor": 0, 
         "geometry": {"start": [8000, 20000, 3000], "end": [14000, 20000, 3000], "width": 300.0}},
        {"id": "slab_f0", "type": "slab", "floor": 0, 
         "geometry": {"start": [2000, 2000, 3000], "end": [18000, 28000, 3000]}},
        {"id": "staircase_001", "type": "staircase", "floor": 0, 
         "geometry": {"start": [10000, 5000, 0], "end": [12000, 8000, 3000], "width": 1200.0}},
        {"id": "opening_f0_001", "type": "opening", "floor": 0, 
         "geometry": {"start": [3000, 2000, 900], "end": [4200, 2000, 2400]}, 
         "properties": {"subtype": "window", "width": 1200.0}},
        {"id": "opening_f0_002", "type": "opening", "floor": 0, 
         "geometry": {"start": [8000, 2000, 0], "end": [8900, 2000, 2100]}, 
         "properties": {"subtype": "door", "width": 900.0}}
    ]
    
    floors = [
        {"level": 0, "rooms": [
            {"name": "Entrance Hall", "area": 60.0, "perimeter": [[2400, 2400], [7880, 2400], [7880, 27600], [2400, 27600]]},
            {"name": "Technical Room", "area": 80.0, "perimeter": [[8120, 2400], [11880, 2400], [11880, 27600], [8120, 27600]]},
            {"name": "Storage", "area": 60.0, "perimeter": [[12120, 2400], [17600, 2400], [17600, 27600], [12120, 27600]]}
        ]}
    ]
    
    mep = {
        "electrical": {"main_panel": [5000, 15000, 0]},
        "plumbing": {"wet_risers": [[15000, 8000], [15000, 22000]]}
    }
    
    _add_layers(doc)
    _draw_grid(msp, building, site)
    _draw_walls(msp, elements)
    _draw_columns(msp, elements)
    _draw_beams(msp, elements)
    _draw_slabs(msp, elements)
    _draw_openings(msp, elements)
    _draw_stairs_elevators(msp, elements)
    _draw_mep(msp, mep)
    _draw_rooms(msp, floors)
    _draw_dimensions(msp, building, site)
    _draw_title_block(msp, building)

    # Set initial view so the drawing is visible when opened.
    # Footprint: X=2000..18000, Y=2000..28000, title block at X=22000..26000
    # center = midpoint of total area; height = total Y span + margin
    doc.set_modelspace_vport(
        height=34000,       # total visible height in mm (includes margins)
        center=(14000, 15000),  # center of drawing including title block
    )

    return doc

if __name__ == "__main__":
    drawing = generate_drawing()
    drawing.saveas("output.dxf")
