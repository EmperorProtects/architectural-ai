"""
Data Collector — captures pipeline outputs and architect feedback,
feeding them into the FeedbackStore for future fine-tuning.
"""

import logging
from typing import Optional, List, Dict, Any
from .models import FeedbackRecord, ElementCorrection
from .feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Sits at the end of the generation pipeline.
    Records every generation + collects architect corrections (Human-in-the-Loop).
    """

    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()

    def record_generation(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        building_type: str,
        generated_ast: Dict[str, Any],
        action_plan: Optional[Dict[str, Any]] = None,
        generated_cad_code: Optional[str] = None,
        cad_target: str = "autocad_ezdxf",
        validation_errors: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
    ) -> str:
        """
        Record a complete generation cycle.
        Returns the record ID for later feedback submission.
        """
        record = FeedbackRecord(
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            building_type=building_type,
            action_plan=action_plan,
            generated_ast=generated_ast,
            generated_cad_code=generated_cad_code,
            cad_target=cad_target,
            validation_errors=validation_errors or [],
            validation_warnings=validation_warnings or [],
        )
        record_id = self.store.save_record(record)
        logger.info(f"Recorded generation → record_id={record_id}")
        return record_id

    def submit_architect_feedback(
        self,
        record_id: str,
        quality_score: int,
        corrected_ast: Optional[Dict[str, Any]] = None,
        element_corrections: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[str] = None,
        approve: bool = True,
    ) -> bool:
        """
        Submit architect review feedback for a generation record.

        Args:
            record_id: ID returned by record_generation()
            quality_score: 1–5 rating
            corrected_ast: Full corrected AST (if architect modified it)
            element_corrections: List of element-level diffs
            notes: Free-text architect notes
            approve: Whether to mark for training use
        """
        corrections = []
        if element_corrections:
            for c in element_corrections:
                try:
                    corrections.append(ElementCorrection(**c))
                except Exception as e:
                    logger.warning(f"Invalid correction format: {e}")

        success = self.store.approve_record(
            record_id=record_id,
            quality_score=quality_score,
            corrected_ast=corrected_ast,
            corrections=corrections,
            notes=notes,
        )

        if success:
            logger.info(
                f"Feedback recorded: record={record_id}, "
                f"score={quality_score}, approved={approve}"
            )
        return success

    def get_stats(self) -> Dict[str, Any]:
        """Return feedback statistics."""
        return self.store.get_stats()

    def get_training_ready_count(self, min_score: int = 3) -> int:
        """How many approved records are ready for training."""
        return len(self.store.get_approved_unused(min_score=min_score))
