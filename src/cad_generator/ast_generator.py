"""
CAD AST Generator — uses Claude to produce a structured JSON AST
from an architectural prompt, grounded in the normative knowledge base.
"""

import json
import logging
from typing import Optional
import anthropic
from ..knowledge_base import format_for_prompt

logger = logging.getLogger(__name__)

AST_SYSTEM_PROMPT = """You are an expert architectural BIM system producing construction-grade JSON ASTs.
Every element must be described to the FINEST DETAIL — material layers, finishes, fixtures, dimensions, specs.

━━━ FULL SCHEMA ━━━

{
  "building": {
    "type": string,
    "floors": integer,
    "total_area": float,          // m²
    "footprint_area": float,      // m² (single floor)
    "structural_system": "frame" | "bearing_walls" | "mixed",
    "column_grid_x": float,       // mm
    "column_grid_y": float,       // mm
    "ceiling_height": float,      // floor-to-floor mm
    "fire_class": string,         // e.g. "REI 60"
    "energy_class": string,       // e.g. "B", "C"
    "seismic_zone": string        // e.g. "MSK-64 Zone 7 (Almaty)"
  },

  "site": {
    "width": float, "length": float,     // mm
    "setback_front": float, "setback_side": float, "setback_rear": float,  // mm
    "orientation": "north"|"south"|"east"|"west",
    "slope_percent": float,
    "soil_type": string           // e.g. "loam bearing 200 kPa"
  },

  "elements": [
    // ── WALLS ──────────────────────────────────────────────────────────────
    {
      "id": "wall_f{floor}_{seq}",
      "type": "wall",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,       // primary structural layer material
      "properties": {
        "subtype": "exterior" | "interior" | "partition" | "shear",
        "load_bearing": boolean,
        "total_thickness_mm": float,
        "layers": [             // outside → inside, all thicknesses sum to total_thickness
          { "name": string, "material": string, "thickness_mm": float,
            "standard": string, "function": "structural"|"insulation"|"vapour_barrier"|"finish"|"cladding" }
        ],
        "fire_rating": string,        // e.g. "EI 60"
        "acoustic_rating_dB": float,  // Rw value
        "u_value": float,             // W/m²K thermal transmittance
        "finish_exterior": { "type": string, "color": string, "standard": string },
        "finish_interior": { "type": string, "color": string, "standard": string }
      }
    },

    // ── COLUMNS ────────────────────────────────────────────────────────────
    {
      "id": "col_{grid_label}_{seq}",
      "type": "column",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "cross_section": string,      // e.g. "400x400", "diameter:500"
        "concrete_grade": string,     // e.g. "B25"
        "rebar_longitudinal": { "count": integer, "diameter_mm": float, "grade": string },
        "rebar_stirrups": { "diameter_mm": float, "spacing_mm": float },
        "fire_protection": string,    // e.g. "plaster 20mm REI90"
        "connection_top": "pinned"|"fixed",
        "connection_bottom": "fixed"|"pinned"
      }
    },

    // ── BEAMS ──────────────────────────────────────────────────────────────
    {
      "id": "beam_f{floor}_{seq}",
      "type": "beam",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "cross_section": string,      // e.g. "300x600"
        "span_mm": float,
        "concrete_grade": string,
        "rebar_top": { "count": integer, "diameter_mm": float },
        "rebar_bottom": { "count": integer, "diameter_mm": float },
        "stirrups": { "diameter_mm": float, "spacing_mm": float },
        "camber_mm": float,
        "direction": "X"|"Y"
      }
    },

    // ── SLABS ──────────────────────────────────────────────────────────────
    {
      "id": "slab_f{floor}",
      "type": "slab",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "slab_type": "flat"|"ribbed"|"hollow_core"|"waffle",
        "structural_thickness_mm": float,
        "total_thickness_mm": float,
        "layers": [             // top → bottom
          { "name": string, "material": string, "thickness_mm": float, "standard": string }
        ],
        "concrete_grade": string,
        "rebar_bottom_x": { "diameter_mm": float, "spacing_mm": float },
        "rebar_bottom_y": { "diameter_mm": float, "spacing_mm": float },
        "rebar_top": { "diameter_mm": float, "spacing_mm": float },
        "fire_rating": string,
        "acoustic_impact_rating_dB": float,
        "waterproofed": boolean
      }
    },

    // ── OPENINGS (windows & doors) ─────────────────────────────────────────
    {
      "id": "opening_f{floor}_{seq}",
      "type": "opening",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "subtype": "window" | "door" | "glazed_door" | "skylight",
        "parent_wall_id": string,
        "clear_width_mm": float,
        "clear_height_mm": float,
        "sill_height_mm": float,       // from finished floor
        "lintel_height_mm": float,     // from finished floor to top of opening
        "frame": {
          "material": string,          // e.g. "aluminium 6063-T5", "PVC 70mm profile"
          "color": string,
          "thermal_break": boolean,
          "standard": string
        },
        "glazing": {
          "type": "single"|"double"|"triple",
          "ug_value": float,           // W/m²K
          "g_value": float,            // solar factor
          "thickness_mm": string,      // e.g. "4-16Ar-4"
          "coating": string            // e.g. "low-e"
        },
        "hardware": { "handle": string, "lock": string, "hinge_count": integer },
        "fire_rating": string,         // null if not fire-rated
        "acoustic_rating_dB": float,
        "opening_direction": "inward"|"outward"|"sliding"|"folding"
      }
    },

    // ── STAIRCASES ─────────────────────────────────────────────────────────
    {
      "id": "stair_{seq}",
      "type": "staircase",
      "floor": integer,             // base floor
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "flights_per_floor": integer,
        "clear_width_mm": float,
        "tread_depth_mm": float,      // e.g. 300
        "riser_height_mm": float,     // e.g. 150
        "treads_per_flight": integer,
        "landing_width_mm": float,
        "handrail": { "height_mm": float, "material": string, "sides": "left"|"right"|"both" },
        "nosing": { "type": string, "material": string },
        "finish": string,
        "fire_escape": boolean,
        "pressurized": boolean        // smoke-pressurised stair shaft
      }
    },

    // ── ELEVATORS ──────────────────────────────────────────────────────────
    {
      "id": "elevator_{seq}",
      "type": "elevator",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": "concrete B25",
      "properties": {
        "shaft_width_mm": float,
        "shaft_depth_mm": float,
        "car_width_mm": float,
        "car_depth_mm": float,
        "capacity_kg": integer,
        "persons": integer,
        "speed_ms": float,
        "floors_served": [integer],
        "drive_type": "traction"|"hydraulic"|"MRL",
        "door_type": "centre_opening"|"side_opening",
        "door_width_mm": float,
        "accessibility": boolean      // wheelchair compliant
      }
    },

    // ── MEP SHAFTS ─────────────────────────────────────────────────────────
    {
      "id": "mep_shaft_{seq}",
      "type": "mep_shaft",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": "concrete B25",
      "properties": {
        "shaft_type": "wet"|"dry"|"combined",
        "services": [string],         // e.g. ["cold_water", "hot_water", "sewage", "heating_supply"]
        "shaft_width_mm": float,
        "shaft_depth_mm": float,
        "access_panel": boolean,
        "fire_stopping": boolean
      }
    },

    // ── BALCONIES ──────────────────────────────────────────────────────────
    {
      "id": "balcony_f{floor}_{seq}",
      "type": "balcony",
      "floor": integer,
      "geometry": { "start": [x,y,z], "end": [x,y,z], "width": float, "height": float },
      "material": string,
      "properties": {
        "depth_mm": float,
        "width_mm": float,
        "slab_thickness_mm": float,
        "railing": { "height_mm": float, "type": "glass"|"metal"|"concrete", "material": string },
        "waterproofing": string,
        "finish": string,
        "drainage": boolean
      }
    }
  ],

  "floors": [
    {
      "level": integer,
      "elevation": float,          // mm
      "height": float,             // floor-to-floor mm
      "function": string,
      "rooms": [
        {
          "id": "room_f{floor}_{seq}",
          "name": string,
          "type": string,
          "area": float,            // m²
          "net_area": float,        // m² excluding walls
          "perimeter": [[x,y],...], // 2-D polygon corners mm
          "boundary_elements": [string],
          "has_window": boolean,
          "has_door": boolean,
          "natural_light": boolean,
          "ceiling_height_mm": float,

          "finishes": {
            "floor": { "type": string, "material": string, "thickness_mm": float,
                       "standard": string, "color": string },
            "walls": { "type": string, "material": string, "thickness_mm": float,
                       "standard": string, "color": string },
            "ceiling": { "type": string, "material": string, "color": string,
                         "suspended": boolean, "plenum_height_mm": float }
          },

          "fixtures": {
            "lighting": [
              { "id": string, "type": "ceiling"|"wall"|"floor"|"recessed",
                "position": [x,y], "wattage_W": float, "lumens": float,
                "color_temp_K": integer, "standard": string }
            ],
            "electrical": [
              { "id": string, "type": "socket_230V"|"socket_400V"|"switch"|"data_RJ45"|"tv_coax",
                "position": [x,y], "height_mm": float, "circuit_id": string }
            ],
            "plumbing": [           // only in wet rooms
              { "id": string, "type": "toilet"|"sink"|"shower"|"bathtub"|"bidet"|"floor_drain"|"washing_machine",
                "position": [x,y], "drain_pipe_dia_mm": float, "supply_pipe_dia_mm": float,
                "standard": string }
            ],
            "hvac": [
              { "id": string, "type": "radiator"|"fan_coil"|"supply_diffuser"|"return_grille"|"fan"|"thermostat",
                "position": [x,y], "height_mm": float,
                "capacity_W": float, "airflow_m3h": float }
            ],
            "fire": [
              { "id": string, "type": "smoke_detector"|"heat_detector"|"sprinkler"|"fire_hydrant"|"emergency_light"|"exit_sign",
                "position": [x,y], "height_mm": float, "coverage_m2": float }
            ]
          }
        }
      ]
    }
  ],

  "mep": {
    "electrical": {
      "main_panel": { "position": [x,y,floor], "capacity_kVA": float, "circuits": integer },
      "sub_panels": [ { "position": [x,y,floor], "floor": integer, "capacity_kVA": float } ],
      "riser_shaft_ids": [string],
      "cable_tray_routes": [ { "from": [x,y,z], "to": [x,y,z], "width_mm": float } ]
    },
    "plumbing": {
      "cold_water_entry": [x,y,z],
      "hot_water_entry": [x,y,z],
      "sewage_exit": [x,y,z],
      "wet_risers": [ { "id": string, "position": [x,y], "pipe_dia_mm": float } ],
      "heating_risers": [ { "id": string, "position": [x,y], "supply_temp_C": float, "return_temp_C": float } ],
      "shaft_ids": [string]
    },
    "hvac": {
      "ahu_room_floor": integer,
      "ahu_room_area": float,
      "ahu_capacity_m3h": float,
      "shaft_ids": [string],
      "duct_routes": [ { "from": [x,y,z], "to": [x,y,z], "size_mm": string } ]
    },
    "fire": {
      "sprinkler_zones": integer,
      "hydrant_count": integer,
      "escape_routes": [string],
      "fire_alarm_zones": integer,
      "suppression_type": "wet_pipe"|"dry_pipe"|"pre_action"
    }
  },

  "metadata": {
    "generated_by": "architectural-ai",
    "norms_applied": [string],
    "warnings": [string],
    "grid_origin": [x, y],
    "total_element_count": integer,
    "total_fixture_count": integer
  }
}

━━━ MANDATORY REQUIREMENTS (AST rejected if violated) ━━━
1. ALL coordinates in mm from (0,0,0). Z = floor elevation.
2. EVERY floor → full room list. EVERY room → finishes + ALL fixture categories (lighting, electrical, hvac, fire; plumbing only in wet rooms).
3. EVERY wall → full layers array (exterior walls min 3 layers, interior min 2).
4. EVERY slab → full layers array top-to-bottom, waterproofed=true for ground floor and bathrooms.
5. EVERY column → rebar_longitudinal + rebar_stirrups.
6. EVERY opening → frame + glazing + hardware objects fully populated.
7. Exterior walls all 4 faces per floor, thickness 300–400mm. Interior partitions enclose every room.
8. Columns at every grid intersection every floor. Beams between every adjacent column pair.
9. Windows every 3–4m on exterior walls; doors on every room.
10. Staircase ≥1 (width ≥1200mm). Elevator if floors ≥5. MEP wet+dry shafts.
11. EVERY room has ≥1 lighting fixture, ≥1 electrical socket, ≥1 HVAC unit, ≥1 fire detector.
12. Wet rooms (bathroom, kitchen, WC) have full plumbing fixtures list.
13. Return ONLY valid JSON. No markdown, no prose."""

ACTION_PLAN_PROMPT = """Based on the architectural prompt and building norms, produce a COMPREHENSIVE ACTION PLAN.
The plan must pre-decide every dimension, material, and fixture count so the AST generator has no ambiguity.

Return ONLY valid JSON:
{
  "building_summary": {
    "type": string, "floors": integer,
    "footprint_w_mm": float, "footprint_d_mm": float,
    "total_area_m2": float, "footprint_area_m2": float,
    "floor_to_floor_mm": float, "structural_ceiling_mm": float,
    "fire_class": string, "seismic_zone": string, "energy_class": string
  },
  "structural": {
    "system": string,
    "column_grid": { "spacing_x_mm": float, "spacing_y_mm": float, "bays_x": integer, "bays_y": integer },
    "column_cross_section_mm": string,
    "beam_cross_section_mm": string,
    "slab_thickness_mm": float,
    "concrete_grade": string,
    "rebar_grade": string
  },
  "wall_specs": {
    "exterior": {
      "total_thickness_mm": float,
      "layers": [{"name": string, "material": string, "thickness_mm": float}],
      "u_value": float, "fire_rating": string, "acoustic_dB": float
    },
    "interior_load_bearing": { "total_thickness_mm": float, "layers": [...] },
    "partition": { "total_thickness_mm": float, "layers": [...] }
  },
  "slab_spec": {
    "layers": [{"name": string, "material": string, "thickness_mm": float}],
    "total_thickness_mm": float, "waterproof_layers": [string]
  },
  "opening_specs": {
    "window": {
      "typical_width_mm": float, "typical_height_mm": float, "sill_height_mm": float,
      "frame_material": string, "glazing": string, "ug_value": float
    },
    "exterior_door": { "width_mm": float, "height_mm": float, "material": string },
    "interior_door": { "width_mm": float, "height_mm": float, "material": string }
  },
  "room_program": {
    "floor_0": [
      { "name": string, "type": string, "area_m2": float, "width_mm": float, "depth_mm": float,
        "windows": integer, "doors": integer,
        "fixtures": { "lighting": integer, "sockets": integer, "hvac_units": integer,
                      "plumbing": [string], "fire_detectors": integer } }
    ],
    "typical_floor": [...],
    "top_floor": [...]
  },
  "mep_layout": {
    "staircase": [{ "position": [x,y], "width_mm": float, "flights": integer }],
    "elevator": [{ "position": [x,y], "shaft_mm": string, "capacity_kg": integer }],
    "wet_shafts": [{ "id": string, "position": [x,y], "services": [string] }],
    "dry_shafts": [{ "id": string, "position": [x,y], "services": [string] }],
    "ahu_floor": integer, "ahu_capacity_m3h": float,
    "electrical_main_panel": [x,y,floor],
    "sub_panel_per_floor": boolean
  },
  "finish_schedule": {
    "living_room": { "floor": string, "walls": string, "ceiling": string },
    "bedroom":     { "floor": string, "walls": string, "ceiling": string },
    "kitchen":     { "floor": string, "walls": string, "ceiling": string },
    "bathroom":    { "floor": string, "walls": string, "ceiling": string },
    "corridor":    { "floor": string, "walls": string, "ceiling": string }
  },
  "element_counts": {
    "exterior_walls": integer, "interior_walls": integer, "partitions": integer,
    "columns": integer, "beams": integer, "slabs": integer,
    "windows": integer, "doors": integer,
    "staircases": integer, "elevators": integer, "mep_shafts": integer, "balconies": integer,
    "total_lighting_fixtures": integer, "total_electrical_outlets": integer,
    "total_plumbing_fixtures": integer, "total_hvac_units": integer, "total_fire_devices": integer
  },
  "norms_applied": [string],
  "key_decisions": [string]
}"""


class ASTGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_action_plan(self, prompt: str, building_type: str = "residential") -> dict:
        """Generate a step-by-step construction plan before building the AST."""
        norms_context = format_for_prompt(building_type)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=ACTION_PLAN_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Architectural prompt:\n{prompt}\n\n{norms_context}",
                }
            ],
        )

        raw = response.content[0].text.strip()
        # Strip any markdown fence
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from mixed output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse action plan JSON, proceeding without it")
            return {"steps": []}

    def generate_ast(
        self,
        prompt: str,
        building_type: str = "residential",
        action_plan: Optional[dict] = None,
    ) -> dict:
        """Generate full JSON AST from architectural prompt."""
        norms_context = format_for_prompt(building_type)

        plan_context = ""
        if action_plan:
            plan_context = f"\n\nAction plan (follow exactly — dimensions, materials, fixture counts are pre-decided):\n{json.dumps(action_plan, ensure_ascii=False, indent=2)}"

        user_message = (
            f"Generate the detailed JSON AST for:\n\n{prompt}\n\n"
            f"{norms_context}{plan_context}\n\n"
            "TOKEN BUDGET STRATEGY — follow exactly to fit within 16000 tokens:\n"
            "1. 'elements': include ALL structural elements for ALL floors (walls, columns, beams, slabs, openings, stairs, elevator, mep_shafts, balconies). Use concise IDs.\n"
            "2. 'floors': include ONLY floor 0 (ground) and floor 1 (typical) with FULL room detail (finishes + fixtures). "
            "   For floors 2+ add entries with level/elevation/height/function but rooms=[] — the builder replicates floor 1.\n"
            "3. In each room keep fixtures concise: 1-2 examples per category, not every single unit.\n"
            "4. Wall layers: max 3 layers per wall. Column rebar: just the counts and diameters.\n"
            "5. Return ONLY valid JSON — ensure it is complete and not truncated."
        )

        raw_chunks = []
        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=AST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                raw_chunks.append(text)

        raw = "".join(raw_chunks).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        raw = raw.strip()
        ast = json.loads(raw)

        # Ensure metadata exists
        ast.setdefault("metadata", {})
        ast["metadata"]["generated_by"] = "architectural-ai/claude"
        ast["metadata"].setdefault("warnings", [])

        # Replicate typical floor rooms to upper floors that have empty rooms lists
        ast = self._replicate_upper_floors(ast)

        logger.info(
            f"AST generated: {len(ast.get('elements', []))} elements, "
            f"{len(ast.get('floors', []))} floors"
        )
        return ast

    def _replicate_upper_floors(self, ast: dict) -> dict:
        """Fill floors with empty rooms by cloning the typical floor room program."""
        floors = ast.get("floors", [])
        if len(floors) < 2:
            return ast

        # Find the typical floor (level 1, or first floor with rooms)
        typical = next(
            (f for f in floors if f.get("level", 0) == 1 and f.get("rooms")),
            next((f for f in floors if f.get("rooms")), None),
        )
        if not typical:
            return ast

        typical_rooms = typical.get("rooms", [])
        typical_elevation = typical.get("elevation", 3000)
        floor_height = typical.get("height", 3000)

        for fl in floors:
            if fl.get("rooms"):
                continue  # already populated
            lvl = fl.get("level", 0)
            elev = fl.get("elevation", lvl * floor_height)
            delta_z = elev - typical_elevation

            # Clone rooms with shifted perimeter Z (perimeter is 2D, so just copy)
            import copy
            cloned = []
            for room in typical_rooms:
                r = copy.deepcopy(room)
                r["id"] = r["id"].replace("_f1_", f"_f{lvl}_").replace("_f0_", f"_f{lvl}_")
                cloned.append(r)
            fl["rooms"] = cloned

        return ast

    def generate_with_plan(self, prompt: str, building_type: str = "residential") -> dict:
        """Full pipeline: action plan → AST."""
        logger.info("Step 1: Generating action plan...")
        plan = self.generate_action_plan(prompt, building_type)

        logger.info("Step 2: Generating JSON AST...")
        ast = self.generate_ast(prompt, building_type, action_plan=plan)

        return {"action_plan": plan, "ast": ast}
