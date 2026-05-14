"""
AST Validator — checks the generated JSON AST for geometric consistency,
structural integrity, and normative compliance before CAD code generation.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_ast: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}): " + "; ".join(self.errors))
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}): " + "; ".join(self.warnings))
        return " | ".join(lines)


class ASTValidator:
    MIN_WALL_THICKNESS = 100      # mm
    MAX_WALL_THICKNESS = 1000     # mm
    MIN_CEILING_HEIGHT = 2400     # mm
    MAX_CEILING_HEIGHT = 20000    # mm
    MIN_COLUMN_SPACING = 3000     # mm
    MAX_SPAN = 18000              # mm

    def validate(self, ast: dict) -> ValidationResult:
        result = ValidationResult(is_valid=True, corrected_ast=ast.copy())

        self._validate_schema(ast, result)
        if not result.is_valid:
            return result  # Don't continue if schema is broken

        self._validate_site(ast.get("site", {}), result)
        self._validate_elements(ast.get("elements", []), result)
        self._validate_floors(ast.get("floors", []), result)
        self._validate_element_ids(ast.get("elements", []), result)
        self._validate_openings(ast.get("elements", []), result)
        self._validate_detail_quality(ast, result)

        # Inject warnings into AST metadata
        if result.warnings:
            result.corrected_ast.setdefault("metadata", {})
            result.corrected_ast["metadata"]["warnings"] = result.warnings

        logger.info(f"AST validation: {result.summary()}")
        return result

    def _validate_schema(self, ast: dict, result: ValidationResult):
        required_keys = ["building", "site", "elements", "floors"]
        for key in required_keys:
            if key not in ast:
                result.add_error(f"Missing required AST key: '{key}'")

        if "building" in ast:
            b = ast["building"]
            if not isinstance(b.get("floors"), int) or b.get("floors", 0) < 1:
                result.add_error("building.floors must be a positive integer")
            if not isinstance(b.get("total_area"), (int, float)) or b.get("total_area", 0) <= 0:
                result.add_error("building.total_area must be a positive number")

    def _validate_site(self, site: dict, result: ValidationResult):
        for dim in ["width", "length"]:
            val = site.get(dim)
            if val is None:
                result.add_error(f"site.{dim} is required")
            elif val <= 0:
                result.add_error(f"site.{dim} must be positive, got {val}")
            elif val < 5000:  # mm
                result.add_warning(f"site.{dim} = {val}mm seems very small (<5m)")

    def _validate_elements(self, elements: List[dict], result: ValidationResult):
        if not elements:
            result.add_error("No elements defined in AST")
            return

        valid_types = {
            "wall", "column", "slab", "beam", "opening",
            "staircase", "elevator", "mep_shaft", "balcony", "ramp",
        }

        for i, el in enumerate(elements):
            el_id = el.get("id", f"element_{i}")

            if el.get("type") not in valid_types:
                result.add_error(f"{el_id}: unknown type '{el.get('type')}'")
                continue

            geom = el.get("geometry", {})
            if not geom:
                result.add_error(f"{el_id}: missing geometry")
                continue

            self._validate_geometry(el_id, el["type"], geom, result)

    def _validate_geometry(
        self, el_id: str, el_type: str, geom: dict, result: ValidationResult
    ):
        start = geom.get("start")
        end = geom.get("end")

        if not (isinstance(start, (list, tuple)) and len(start) == 3):
            result.add_error(f"{el_id}: geometry.start must be [x, y, z]")
            return
        if not (isinstance(end, (list, tuple)) and len(end) == 3):
            result.add_error(f"{el_id}: geometry.end must be [x, y, z]")
            return

        # Check for zero-length elements
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])
        length = max(dx, dy, dz)

        if length < 1:
            result.add_error(f"{el_id}: zero-length element (start == end)")

        # Wall-specific checks
        if el_type == "wall":
            width = geom.get("width", 0)
            height = geom.get("height", 0)

            if not (self.MIN_WALL_THICKNESS <= width <= self.MAX_WALL_THICKNESS):
                result.add_warning(
                    f"{el_id}: wall width {width}mm outside typical range "
                    f"({self.MIN_WALL_THICKNESS}–{self.MAX_WALL_THICKNESS}mm)"
                )

            if not (self.MIN_CEILING_HEIGHT <= height <= self.MAX_CEILING_HEIGHT):
                result.add_error(
                    f"{el_id}: wall height {height}mm outside valid range "
                    f"({self.MIN_CEILING_HEIGHT}–{self.MAX_CEILING_HEIGHT}mm)"
                )

        # Span check applies to beams only — slabs cover the full floor footprint by design
        if length > self.MAX_SPAN and el_type == "beam":
            result.add_warning(
                f"{el_id}: beam span of {length/1000:.1f}m exceeds typical max ({self.MAX_SPAN/1000}m)"
            )

    def _validate_floors(self, floors: List[dict], result: ValidationResult):
        if not floors:
            result.add_warning("No floor definitions found")
            return

        elevations = sorted(f.get("elevation", 0) for f in floors)
        prev = None
        for i, elev in enumerate(elevations):
            if prev is not None and elev - prev < self.MIN_CEILING_HEIGHT:
                result.add_error(
                    f"Floor {i}: elevation gap {elev - prev}mm is less than "
                    f"minimum ceiling height {self.MIN_CEILING_HEIGHT}mm"
                )
            prev = elev

    def _validate_element_ids(self, elements: List[dict], result: ValidationResult):
        seen_ids = set()
        for el in elements:
            el_id = el.get("id")
            if not el_id:
                result.add_error("Element missing 'id' field")
            elif el_id in seen_ids:
                result.add_error(f"Duplicate element id: '{el_id}'")
            else:
                seen_ids.add(el_id)

    def _validate_detail_quality(self, ast: dict, result: ValidationResult):
        """Warn (not error) when fine-detail sections are missing."""
        elements = ast.get("elements", [])
        floors = ast.get("floors", [])

        # Wall layer stacks
        walls = [e for e in elements if e.get("type") == "wall"]
        walls_without_layers = [
            e["id"] for e in walls
            if not e.get("properties", {}).get("layers")
        ]
        if walls_without_layers:
            result.add_warning(
                f"{len(walls_without_layers)} wall(s) missing layer stack: "
                f"{', '.join(walls_without_layers[:3])}{'...' if len(walls_without_layers) > 3 else ''}"
            )

        # Slab layer stacks
        slabs = [e for e in elements if e.get("type") == "slab"]
        slabs_without_layers = [e["id"] for e in slabs if not e.get("properties", {}).get("layers")]
        if slabs_without_layers:
            result.add_warning(f"{len(slabs_without_layers)} slab(s) missing layer stack")

        # Room fixtures and finishes
        total_rooms = 0
        rooms_without_fixtures = []
        rooms_without_finishes = []
        for floor in floors:
            for room in floor.get("rooms", []):
                total_rooms += 1
                fixtures = room.get("fixtures", {})
                if not fixtures.get("lighting"):
                    rooms_without_fixtures.append(room.get("id", "?"))
                if not room.get("finishes"):
                    rooms_without_finishes.append(room.get("id", "?"))

        if rooms_without_fixtures:
            result.add_warning(
                f"{len(rooms_without_fixtures)}/{total_rooms} room(s) missing fixture details"
            )
        if rooms_without_finishes:
            result.add_warning(
                f"{len(rooms_without_finishes)}/{total_rooms} room(s) missing finish schedule"
            )

        # Column rebar
        columns = [e for e in elements if e.get("type") == "column"]
        cols_no_rebar = [
            e["id"] for e in columns
            if not e.get("properties", {}).get("rebar_longitudinal")
        ]
        if cols_no_rebar:
            result.add_warning(f"{len(cols_no_rebar)} column(s) missing rebar specification")

        # Opening specs
        openings = [e for e in elements if e.get("type") == "opening"]
        openings_no_frame = [
            e["id"] for e in openings
            if not e.get("properties", {}).get("frame")
        ]
        if openings_no_frame:
            result.add_warning(f"{len(openings_no_frame)} opening(s) missing frame/glazing spec")

        # Log summary
        fixture_count = sum(
            sum(len(v) for v in room.get("fixtures", {}).values() if isinstance(v, list))
            for floor in floors for room in floor.get("rooms", [])
        )
        logger.info(
            f"Detail quality: {len(elements)} elements, {total_rooms} rooms, "
            f"~{fixture_count} fixtures"
        )

    def _validate_openings(self, elements: List[dict], result: ValidationResult):
        """Openings should reference a valid parent wall (warning only if missing)."""
        wall_ids = {el["id"] for el in elements if el.get("type") == "wall"}
        for el in elements:
            if el.get("type") == "opening":
                props = el.get("properties", {})
                # Accept either field name used by Claude
                parent = props.get("parent_wall_id") or props.get("parent_wall")
                if not parent:
                    result.add_warning(f"{el.get('id')}: opening has no parent_wall_id reference")
                elif parent not in wall_ids:
                    result.add_warning(
                        f"{el.get('id')}: parent_wall_id '{parent}' not found — may be on another floor"
                    )
