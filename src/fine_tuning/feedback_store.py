"""
Feedback Store — SQLite-backed persistence for architect feedback records.
Stores prompt→AST→correction pairs that feed the fine-tuning pipeline.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from .models import FeedbackRecord, ElementCorrection, TrainingBatch

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "feedback.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_records (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    enhanced_prompt TEXT NOT NULL,
    building_type TEXT NOT NULL,
    action_plan TEXT,
    generated_ast TEXT NOT NULL,
    generated_cad_code TEXT,
    cad_target TEXT DEFAULT 'autocad_ezdxf',
    validation_errors TEXT DEFAULT '[]',
    validation_warnings TEXT DEFAULT '[]',
    quality_score INTEGER,
    corrections TEXT DEFAULT '[]',
    corrected_ast TEXT,
    architect_notes TEXT,
    approved INTEGER DEFAULT 0,
    used_in_training INTEGER DEFAULT 0,
    training_batch_id TEXT
);

CREATE TABLE IF NOT EXISTS training_batches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    record_ids TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    avg_quality_score REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    model_checkpoint TEXT,
    training_loss REAL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_approved ON feedback_records(approved);
CREATE INDEX IF NOT EXISTS idx_feedback_used ON feedback_records(used_in_training);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_records(building_type);
"""


class FeedbackStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        logger.info(f"FeedbackStore initialized at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write operations ─────────────────────────────────────────────────

    def save_record(self, record: FeedbackRecord) -> str:
        """Insert or update a feedback record."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback_records VALUES (
                    :id, :timestamp, :original_prompt, :enhanced_prompt,
                    :building_type, :action_plan, :generated_ast,
                    :generated_cad_code, :cad_target,
                    :validation_errors, :validation_warnings,
                    :quality_score, :corrections, :corrected_ast,
                    :architect_notes, :approved, :used_in_training, :training_batch_id
                )
                """,
                {
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "original_prompt": record.original_prompt,
                    "enhanced_prompt": record.enhanced_prompt,
                    "building_type": record.building_type,
                    "action_plan": json.dumps(record.action_plan, ensure_ascii=False),
                    "generated_ast": json.dumps(record.generated_ast, ensure_ascii=False),
                    "generated_cad_code": record.generated_cad_code,
                    "cad_target": record.cad_target,
                    "validation_errors": json.dumps(record.validation_errors),
                    "validation_warnings": json.dumps(record.validation_warnings),
                    "quality_score": record.quality_score,
                    "corrections": json.dumps(
                        [c.model_dump() for c in record.corrections], ensure_ascii=False
                    ),
                    "corrected_ast": json.dumps(record.corrected_ast, ensure_ascii=False)
                    if record.corrected_ast else None,
                    "architect_notes": record.architect_notes,
                    "approved": int(record.approved),
                    "used_in_training": int(record.used_in_training),
                    "training_batch_id": record.training_batch_id,
                },
            )
        logger.info(f"Saved feedback record {record.id}")
        return record.id

    def approve_record(
        self,
        record_id: str,
        quality_score: int,
        corrected_ast: Optional[dict] = None,
        corrections: Optional[List[ElementCorrection]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Approve a record with architect feedback."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE feedback_records SET
                    approved = 1,
                    quality_score = ?,
                    corrected_ast = ?,
                    corrections = ?,
                    architect_notes = ?
                WHERE id = ?
                """,
                (
                    quality_score,
                    json.dumps(corrected_ast, ensure_ascii=False) if corrected_ast else None,
                    json.dumps([c.model_dump() for c in corrections], ensure_ascii=False)
                    if corrections else "[]",
                    notes,
                    record_id,
                ),
            )
            updated = conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        logger.info(f"Approved record {record_id} with score {quality_score}")
        return updated > 0

    def mark_used_in_training(self, record_ids: List[str], batch_id: str):
        """Mark records as used in a training batch."""
        with self._conn() as conn:
            conn.executemany(
                "UPDATE feedback_records SET used_in_training = 1, training_batch_id = ? WHERE id = ?",
                [(batch_id, rid) for rid in record_ids],
            )

    def save_training_batch(self, batch: TrainingBatch):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO training_batches VALUES (
                    :id, :created_at, :record_ids, :record_count,
                    :avg_quality_score, :status, :model_checkpoint,
                    :training_loss, :notes
                )""",
                {
                    "id": batch.id,
                    "created_at": batch.created_at.isoformat(),
                    "record_ids": json.dumps(batch.record_ids),
                    "record_count": batch.record_count,
                    "avg_quality_score": batch.avg_quality_score,
                    "status": batch.status,
                    "model_checkpoint": batch.model_checkpoint,
                    "training_loss": batch.training_loss,
                    "notes": batch.notes,
                },
            )

    # ── Read operations ──────────────────────────────────────────────────

    def get_record(self, record_id: str) -> Optional[FeedbackRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM feedback_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_approved_unused(self, min_score: int = 3, limit: int = 100) -> List[FeedbackRecord]:
        """Fetch approved records not yet used in training."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM feedback_records
                   WHERE approved = 1 AND used_in_training = 0
                     AND (quality_score IS NULL OR quality_score >= ?)
                   ORDER BY timestamp DESC LIMIT ?""",
                (min_score, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback_records").fetchone()[0]
            approved = conn.execute(
                "SELECT COUNT(*) FROM feedback_records WHERE approved = 1"
            ).fetchone()[0]
            trained = conn.execute(
                "SELECT COUNT(*) FROM feedback_records WHERE used_in_training = 1"
            ).fetchone()[0]
            avg_score = conn.execute(
                "SELECT AVG(quality_score) FROM feedback_records WHERE quality_score IS NOT NULL"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT building_type, COUNT(*) FROM feedback_records GROUP BY building_type"
            ).fetchall()
        return {
            "total": total,
            "approved": approved,
            "trained": trained,
            "pending_approval": total - approved,
            "pending_training": approved - trained,
            "avg_quality_score": round(avg_score, 2) if avg_score else None,
            "by_building_type": {row[0]: row[1] for row in by_type},
        }

    def _row_to_record(self, row) -> FeedbackRecord:
        d = dict(row)
        corrections_raw = json.loads(d.get("corrections") or "[]")
        return FeedbackRecord(
            id=d["id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            original_prompt=d["original_prompt"],
            enhanced_prompt=d["enhanced_prompt"],
            building_type=d["building_type"],
            action_plan=json.loads(d["action_plan"]) if d.get("action_plan") else None,
            generated_ast=json.loads(d["generated_ast"]),
            generated_cad_code=d.get("generated_cad_code"),
            cad_target=d.get("cad_target", "autocad_ezdxf"),
            validation_errors=json.loads(d.get("validation_errors") or "[]"),
            validation_warnings=json.loads(d.get("validation_warnings") or "[]"),
            quality_score=d.get("quality_score"),
            corrections=[ElementCorrection(**c) for c in corrections_raw],
            corrected_ast=json.loads(d["corrected_ast"]) if d.get("corrected_ast") else None,
            architect_notes=d.get("architect_notes"),
            approved=bool(d.get("approved", 0)),
            used_in_training=bool(d.get("used_in_training", 0)),
            training_batch_id=d.get("training_batch_id"),
        )
