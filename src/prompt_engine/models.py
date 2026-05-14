from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class BuildingType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    PUBLIC = "public"
    MIXED = "mixed"


class ArchitecturalParams(BaseModel):
    building_type: Optional[BuildingType] = None
    floors: Optional[int] = Field(None, ge=1, le=200)
    total_area: Optional[float] = Field(None, gt=0)
    plot_width: Optional[float] = Field(None, gt=0)
    plot_length: Optional[float] = Field(None, gt=0)
    style: Optional[str] = None
    materials: Optional[List[str]] = None
    ceiling_height: Optional[float] = Field(None, gt=0)
    special_requirements: Optional[List[str]] = None

    # REQUIRED_PARAMS: ClassVar = [
    #     "building_type", "floors", "total_area",
    #     "plot_width", "plot_length"
    # ]
    #
    # def get_missing(self) -> List[str]:
    #     missing = []
    #     for param in self.REQUIRED_PARAMS:
    #         if getattr(self, param) is None:
    #             missing.append(param)
    #     return missing


class EnhancedPrompt(BaseModel):
    original: str
    enhanced_variants: List[str]
    missing_params: List[str] = []
    params: Optional[ArchitecturalParams] = None
    selected_variant: Optional[str] = None

    def select_variant(self, index: int = 0) -> "EnhancedPrompt":
        if 0 <= index < len(self.enhanced_variants):
            self.selected_variant = self.enhanced_variants[index]
        return self

    @property
    def is_ready(self) -> bool:
        return len(self.missing_params) == 0 and self.selected_variant is not None
