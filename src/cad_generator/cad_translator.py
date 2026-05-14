"""
CAD Translator — uses Claude to translate a validated JSON AST
into Python code for Revit API or AutoCAD (pyautocad / ezdxf).
"""

import json
import logging
from typing import Optional, Literal
import anthropic

logger = logging.getLogger(__name__)

CAD_TARGET = Literal["revit", "autocad_ezdxf", "autocad_lisp"]

REVIT_SYSTEM_PROMPT = """You are an expert Autodesk Revit API developer (Python, pyRevit).

Generate Python code that creates the building elements from the JSON AST using the Revit API.

Rules:
1. Use `DB` namespace from `Autodesk.Revit.DB`
2. All coordinates: convert mm -> feet (divide by 304.8)
3. Wrap everything in a `create_building(doc, uidoc)` function
4. Use transactions properly: `with Transaction(doc, 'Create Building') as t:`
5. Handle walls with `Wall.Create()`, columns with `FamilyInstance`, slabs with `Floor.Create()`
6. Add comments for each major section
7. Return ONLY working Python code, no markdown fences

Structure:
import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')
from Autodesk.Revit.DB import *
from Autodesk.Revit.DB.Structure import StructuralType

def create_building(doc, uidoc):
    level = FilteredElementCollector(doc).OfClass(Level).FirstElement()
    with Transaction(doc, 'Generate Building') as t:
        t.Start()
        # ... elements ...
        t.Commit()"""

AUTOCAD_SYSTEM_PROMPT = """You are an expert AutoCAD developer using the ezdxf Python library (>= 1.3).

Generate COMPLETE, PRODUCTION-QUALITY Python code that produces a detailed architectural floor plan from the JSON AST.

DRAWING REQUIREMENTS (all must be present):
1. Column grid lines (GRID layer) — full X and Y grid lines through the footprint with grid labels
   (A, B, C... along X; 1, 2, 3... along Y) at both ends of each line.
2. All structural elements with correct geometry:
   - Walls: closed lwpolyline with correct thickness + ANSI31 hatch on HATCH_WALLS layer (scale=30)
   - Columns: solid filled square + SOLID hatch on HATCH_COLS layer
   - Beams: double-line rectangle between columns on BEAMS layer
   - Slabs: boundary polyline + cross-hatch on SLABS layer
3. Openings:
   - Windows: gap in wall polyline + three parallel lines (glazing) on OPENINGS layer
   - Doors: gap in wall + arc sweep + door leaf line on OPENINGS layer
4. Room labels (ROOMS layer): for every room, centered MTEXT showing "Room Name\\nXX.X m2"
5. Dimensions (DIMS layer):
   - Overall building width and depth, offset 1500 mm outside the building
   - Column grid spacing along both axes
   - Every call uses add_aligned_dim(..., distance=<mm>, ...) then .render()
6. MEP elements (MEP layer): labeled rectangles for HVAC shafts, wet shafts, electrical panels
7. Staircases (STAIRS layer): flight lines + tread lines every 250 mm + direction arrow + UP/DN label
8. Elevators (ELEVATOR layer): shaft rectangle + center X mark + "LIFT" label
9. Title block (TITLE_BLOCK layer, bottom-right, outside the footprint):
   Project name, scale "1:100", north arrow triangle, drawing number

Layers to create before first use (doc.layers.add(name=..., color=...)):
  GRID(8), WALLS(1), COLUMNS(1), BEAMS(5), SLABS(3), OPENINGS(4),
  STAIRS(6), ELEVATOR(6), MEP(2), HATCH_WALLS(8), HATCH_COLS(254),
  ROOMS(7), ANNOTATIONS(7), DIMS(7), TITLE_BLOCK(7)

CRITICAL ezdxf API rules (>= 1.0):
- set_pos() is REMOVED. Position TEXT via dxfattribs={'insert': (x, y)}, or call
  entity.set_placement((x, y), align=TextEntityAlignment.MIDDLE_CENTER).
- TEXT is ASCII-only. Use add_mtext() for ALL labels, annotations, and title block text.
  mt = msp.add_mtext("Room\\n24.5 m2", dxfattribs={"layer": "ROOMS", "char_height": 200, "insert": (cx, cy)})
  mt.set_location((cx, cy), attachment_point=5)   # 5 = MIDDLE_CENTER
- add_aligned_dim ALWAYS requires the `distance` argument (offset in mm). Always call .render():
  dim = msp.add_aligned_dim(p1=(x1,y1), p2=(x2,y2), distance=800, dimstyle="EZDXF",
                             override={"dimtxt": 250}, dxfattribs={"layer": "DIMS"})
  dim.render()
- Never pass a literal number string to text= in dim calls — omit or use "<>".
- Polylines: msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "WALLS"})
- Hatch:
  hatch = msp.add_hatch(dxfattribs={"layer": "HATCH_WALLS"})
  hatch.set_pattern_fill("ANSI31", scale=30)
  hatch.paths.add_polyline_path(pts, is_closed=True)
- ezdxf.new("R2010", setup=True) — always use setup=True for EZDXF dimstyle.
- doc.header["$INSUNITS"] = 4  # 4 = millimeters — always set this so viewers scale correctly.
- Before returning doc, set the initial viewport so the drawing is visible when opened.
  DO NOT use zoom.extents() — it fails silently for MTEXT/DIMENSION entities.
  Instead call doc.set_modelspace_vport() with values derived from the site footprint:
    site_w = ast["site"]["width"]   # mm
    site_l = ast["site"]["length"]  # mm
    cx = site_w / 2 + 4000   # +4000 shifts right to include title block
    cy = site_l / 2
    doc.set_modelspace_vport(height=site_l + 12000, center=(cx, cy))
  Without this the DXF opens as a blank white screen.
- doc.saveas("output.dxf") — never doc.save() without a prior filename.

Imports required at top of generated file:
  import ezdxf
  from ezdxf import zoom
  from ezdxf.enums import TextEntityAlignment
  import math

Code must be split into helper functions:
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

  def generate_drawing() -> ezdxf.document.Drawing:
      doc = ezdxf.new("R2010", setup=True)
      msp = doc.modelspace()
      _add_layers(doc)
      ...
      return doc

Return ONLY working Python code — no markdown fences, no prose."""

SYSTEM_PROMPTS = {
    "revit": REVIT_SYSTEM_PROMPT,
    "autocad_ezdxf": AUTOCAD_SYSTEM_PROMPT,
    "autocad_lisp": AUTOCAD_SYSTEM_PROMPT,
}


class CADTranslator:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def translate(
        self,
        ast: dict,
        target: CAD_TARGET = "autocad_ezdxf",
        additional_context: str = "",
    ) -> str:
        """Translate JSON AST to CAD code for the specified target platform."""
        system_prompt = SYSTEM_PROMPTS.get(target, AUTOCAD_SYSTEM_PROMPT)

        element_count = len(ast.get("elements", []))
        floor_count = len(ast.get("floors", []))
        warnings = ast.get("metadata", {}).get("warnings", [])

        user_message = (
            f"Generate {target} code for this building AST.\n\n"
            f"Summary: {element_count} elements, {floor_count} floors\n"
        )
        if warnings:
            user_message += f"AST warnings: {'; '.join(warnings)}\n"
        if additional_context:
            user_message += f"\nAdditional context: {additional_context}\n"

        user_message += f"\nFull AST:\n{json.dumps(ast, ensure_ascii=False, indent=2)}"

        logger.info(f"Translating AST to {target} code ({element_count} elements)...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        code = response.content[0].text.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        logger.info(f"CAD code generated: {len(code.splitlines())} lines")
        return code

    def translate_incremental(
        self,
        ast: dict,
        target: CAD_TARGET = "autocad_ezdxf",
        chunk_size: int = 20,
    ) -> str:
        """For large ASTs: translate in chunks and merge."""
        elements = ast.get("elements", [])
        if len(elements) <= chunk_size:
            return self.translate(ast, target)

        logger.info(f"Large AST ({len(elements)} elements) — using incremental translation")
        chunks = [elements[i:i+chunk_size] for i in range(0, len(elements), chunk_size)]
        code_parts = []

        for idx, chunk in enumerate(chunks):
            chunk_ast = {**ast, "elements": chunk}
            chunk_code = self.translate(
                chunk_ast, target,
                additional_context=f"Chunk {idx+1}/{len(chunks)} — append to existing drawing",
            )
            code_parts.append(f"# chunk {idx+1}/{len(chunks)}\n{chunk_code}")

        return "\n\n".join(code_parts)
