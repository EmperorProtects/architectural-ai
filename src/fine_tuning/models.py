"""
Data models for the fine-tuning feedback loop.
Human-in-the-Loop: architect reviews → corrections stored → model improves.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import uuid


class CorrectionType(str):
    GEOMETRY = "geometry"
    DIMENSIONS = "dimensions"
    MATERIAL = "material"
    ROOM_LAYOUT = "room_layout"
    STRUCTURAL = "structural"
    NORMATIVE = "normative"
    OTHER = "other"


class ElementCorrection(BaseModel):
    """A single correction made by an architect to a specific element."""
    element_id: str
    correction_type: str
    field: str                    # which field was corrected (e.g., "geometry.width")
    original_value: Any
    corrected_value: Any
    reason: Optional[str] = None


class FeedbackRecord(BaseModel):
    """Complete feedback record for one generation cycle."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Input
    original_prompt: str
    enhanced_prompt: str
    building_type: str

    # Generation outputs
    action_plan: Optional[Dict[str, Any]] = None
    generated_ast: Dict[str, Any] = Field(...)
    generated_cad_code: Optional[str] = None
    cad_target: str = "autocad_ezdxf"

    # Validation results
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    # Architect feedback
    quality_score: Optional[int] = Field(None, ge=1, le=5)   # 1–5 stars
    corrections: List[ElementCorrection] = []
    corrected_ast: Optional[Dict[str, Any]] = None
    architect_notes: Optional[str] = None
    approved: bool = False

    # Fine-tuning metadata
    used_in_training: bool = False
    training_batch_id: Optional[str] = None

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0 or self.corrected_ast is not None

    @property
    def training_pair(self) -> Optional[Dict[str, Any]]:
        """Return (input, output) pair for fine-tuning if record is approved."""
        if not self.approved or not self.corrected_ast:
            return None
        return {
            "input": {
                "prompt": self.enhanced_prompt,
                "building_type": self.building_type,
            },
            "output": self.corrected_ast,
            "quality_score": self.quality_score,
        }


class TrainingBatch(BaseModel):
    """A batch of approved feedback records prepared for fine-tuning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    record_ids: List[str]
    record_count: int
    avg_quality_score: float
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    model_checkpoint: Optional[str] = None
    training_loss: Optional[float] = None
    notes: Optional[str] = None
