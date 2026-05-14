"""
Normative knowledge base — building codes and architectural standards.
Used by Claude to ground generation in real regulations (СНиП, ГОСТ, SP).
"""

BUILDING_NORMS = {
    "residential": {
        "min_ceiling_height": 2.5,
        "floor_to_floor_height_mm": 3000,
        "min_room_area": {"bedroom": 8.0, "living": 12.0, "kitchen": 6.0, "bathroom": 3.0, "corridor": 1.4},
        "max_room_area": {"bedroom": 25.0, "living": 40.0, "kitchen": 20.0},
        "exterior_wall_thickness_mm": 400,
        "interior_wall_thickness_mm": 200,
        "partition_thickness_mm": 120,
        "fire_resistance": "REI 60",
        "codes": ["СП 54.13330.2022", "ГОСТ 21.501-2018", "СП 1.13130.2020"],
        "typical_floor_height": 3.0,
        "load_bearing_wall_thickness": 0.2,
        "insulation_required": True,
        "window_height_mm": 1500,
        "window_sill_height_mm": 900,
        "door_width_mm": 900,
        "door_height_mm": 2100,
        "balcony_depth_mm": 1500,
        "stair_width_mm": 1200,
        "stair_tread_mm": 300,
        "stair_riser_mm": 150,
        "elevator_required_above_floors": 5,
        "elevator_shaft_mm": "1600x2100",
        "room_program_per_floor": [
            "living_room", "bedroom_1", "bedroom_2", "kitchen", "bathroom", "toilet", "corridor", "balcony"
        ],
    },
    "commercial": {
        "min_ceiling_height": 3.0,
        "floor_to_floor_height_mm": 4000,
        "exterior_wall_thickness_mm": 300,
        "interior_wall_thickness_mm": 200,
        "fire_resistance": "REI 90",
        "codes": ["СП 118.13330.2022", "СНиП 21-02-99", "ГОСТ Р 51631-2008"],
        "typical_floor_height": 4.0,
        "load_bearing_wall_thickness": 0.25,
        "elevator_required_above_floors": 3,
        "window_height_mm": 2000,
        "window_sill_height_mm": 600,
        "door_width_mm": 1200,
        "stair_width_mm": 1500,
        "room_program_per_floor": [
            "open_office", "meeting_room", "reception", "server_room", "storage", "restrooms", "lobby"
        ],
    },
    "industrial": {
        "min_ceiling_height": 4.0,
        "floor_to_floor_height_mm": 6000,
        "exterior_wall_thickness_mm": 300,
        "fire_resistance": "REI 120",
        "codes": ["СП 56.13330.2021", "СНиП 31-03-2001"],
        "typical_floor_height": 6.0,
        "crane_beam_reserve": True,
        "column_grid_mm": "6000x12000",
        "room_program_per_floor": [
            "production_hall", "storage", "loading_dock", "control_room", "utility_room", "sanitary_zone"
        ],
    },
    "public": {
        "min_ceiling_height": 3.0,
        "floor_to_floor_height_mm": 3600,
        "exterior_wall_thickness_mm": 400,
        "interior_wall_thickness_mm": 200,
        "accessibility": "ГОСТ Р 51631-2008",
        "fire_resistance": "REI 90",
        "codes": ["СП 59.13330.2020", "СП 1.13130.2020", "ГОСТ 21.501-2018"],
        "typical_floor_height": 3.6,
        "ramp_required": True,
        "ramp_width_mm": 1800,
        "room_program_per_floor": [
            "hall", "auditorium", "office", "restrooms", "storage", "technical_room", "lobby"
        ],
    },
}

STRUCTURAL_SYSTEMS = {
    "frame": {
        "description": "Reinforced concrete or steel frame with infill walls",
        "typical_span": "6–12m",
        "suitable_for": ["commercial", "industrial", "public"],
        "column_grid": ["6x6", "6x9", "6x12"],
    },
    "bearing_walls": {
        "description": "Load-bearing masonry or concrete walls",
        "typical_span": "3–6m",
        "suitable_for": ["residential"],
        "wall_thickness": "200–400mm",
    },
    "mixed": {
        "description": "Combination of frame and bearing walls",
        "suitable_for": ["residential", "commercial", "mixed"],
    },
}

MATERIAL_LIBRARY = {
    "concrete": {"grade": ["B20", "B25", "B30", "B40"], "standard": "ГОСТ 26633-2015"},
    "brick": {"grade": ["M100", "M150", "M200"], "standard": "ГОСТ 530-2012"},
    "steel": {"grade": ["С235", "С345", "С390"], "standard": "ГОСТ 27772-2015"},
    "timber": {"grade": ["C16", "C24", "C35"], "standard": "ГОСТ 8486-86"},
    "glass": {"types": ["float", "tempered", "laminated"], "standard": "ГОСТ 111-2014"},
}

MEP_ZONES = {
    "electrical": ["main_panel", "distribution_panels", "cable_runs"],
    "plumbing": ["risers", "horizontal_runs", "fixtures"],
    "hvac": ["ahu_room", "duct_shafts", "fan_coil_units"],
    "fire_protection": ["sprinklers", "fire_hydrants", "smoke_detectors"],
}


def get_norms_for_building(building_type: str) -> dict:
    """Return applicable norms for a building type."""
    return BUILDING_NORMS.get(building_type, BUILDING_NORMS["residential"])


def get_structural_system(system_name: str) -> dict:
    """Return structural system specs."""
    return STRUCTURAL_SYSTEMS.get(system_name, STRUCTURAL_SYSTEMS["frame"])


def format_for_prompt(building_type: str) -> str:
    """Format norms as a string for injection into LLM prompt."""
    norms = get_norms_for_building(building_type)
    lines = [f"=== Building norms for {building_type} ==="]
    for key, value in norms.items():
        lines.append(f"- {key}: {value}")
    lines.append("\nApplicable codes: " + ", ".join(norms.get("codes", [])))

    structural = STRUCTURAL_SYSTEMS.get(
        "frame" if building_type in ("commercial", "industrial", "public") else "mixed"
    )
    lines.append(f"\nTypical structural system: {structural['description']}")
    lines.append(f"Typical span: {structural.get('typical_span', 'N/A')}")
    lines.append(f"Column grid options: {', '.join(structural.get('column_grid', ['6x6m']))}")

    lines.append("\n=== Material library ===")
    for mat, info in MATERIAL_LIBRARY.items():
        grades = info.get("grade") or info.get("types", [])
        lines.append(f"- {mat}: grades {grades}, standard {info.get('standard', '')}")

    lines.append("\n=== Required MEP zones ===")
    for system, components in MEP_ZONES.items():
        lines.append(f"- {system}: {', '.join(components)}")

    return "\n".join(lines)
