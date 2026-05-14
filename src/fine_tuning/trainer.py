"""
Fine-Tuning Trainer — LoRA/QLoRA pipeline for improving the AST generation model.

Architecture:
  Approved feedback records → JSONL dataset → LoRA fine-tune on base model
  (e.g., Llama 3 / Mistral via Ollama, or a dedicated architectural model)

This module is PREPARED but intentionally not connected to the main pipeline yet.
It will be activated once sufficient training data (≥100 approved records) is collected.

Dependencies (install when ready):
    pip install transformers datasets peft trl accelerate bitsandbytes
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from .models import FeedbackRecord, TrainingBatch
from .feedback_store import FeedbackStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"


class DatasetBuilder:
    """Converts approved feedback records into fine-tuning datasets."""

    PROMPT_TEMPLATE = """<|system|>
You are an expert architectural CAD system. Generate a precise JSON AST for the given building prompt.
Apply building norms: {building_type} standards.
<|user|>
{prompt}
<|assistant|>
{corrected_ast}"""

    def __init__(self, output_dir: Path = TRAINING_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_from_records(
        self,
        records: List[FeedbackRecord],
        min_score: int = 3,
        split_ratio: float = 0.9,
    ) -> Dict[str, Path]:
        """
        Build train/validation JSONL files from approved feedback records.

        Format: one JSON object per line (instruction-tuning format).
        Compatible with: HuggingFace TRL, Axolotl, LLaMA-Factory.
        """
        pairs = []
        for record in records:
            pair = record.training_pair
            if pair is None:
                continue
            if record.quality_score and record.quality_score < min_score:
                continue

            # Format as instruction-tuning pair
            formatted = {
                "text": self.PROMPT_TEMPLATE.format(
                    building_type=record.building_type,
                    prompt=record.enhanced_prompt,
                    corrected_ast=json.dumps(
                        record.corrected_ast, ensure_ascii=False, indent=2
                    ),
                ),
                "metadata": {
                    "record_id": record.id,
                    "quality_score": record.quality_score,
                    "building_type": record.building_type,
                },
            }
            pairs.append(formatted)

        if not pairs:
            raise ValueError("No valid training pairs found in records")

        # Train / val split
        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        self._write_jsonl(train_pairs, train_path)
        self._write_jsonl(val_pairs, val_path)

        logger.info(
            f"Dataset built: {len(train_pairs)} train, {len(val_pairs)} val → {self.output_dir}"
        )
        return {"train": train_path, "val": val_path}

    def _write_jsonl(self, records: List[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LoRATrainer:
    """
    LoRA fine-tuning pipeline.
    Uses PEFT + TRL SFTTrainer for parameter-efficient fine-tuning.

    Status: READY — activate when ≥100 approved records are collected.
    """

    DEFAULT_CONFIG = {
        "base_model": "mistralai/Mistral-7B-v0.1",   # or local Ollama model path
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "warmup_ratio": 0.1,
        "save_steps": 50,
        "eval_steps": 50,
        "load_in_4bit": True,     # QLoRA
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Path = CHECKPOINTS_DIR,
    ):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required training libraries are installed."""
        deps = {}
        for lib in ["transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets"]:
            try:
                __import__(lib)
                deps[lib] = True
            except ImportError:
                deps[lib] = False
        return deps

    def train(
        self,
        train_path: Path,
        val_path: Path,
        batch_id: str,
        store: Optional[FeedbackStore] = None,
    ) -> Dict[str, Any]:
        """
        Run LoRA fine-tuning.
        Requires: transformers, peft, trl, accelerate, bitsandbytes
        """
        deps = self.check_dependencies()
        missing = [k for k, v in deps.items() if not v]
        if missing:
            raise ImportError(
                f"Missing training dependencies: {missing}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

        # ── Import training libraries ────────────────────────────────────
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset

        logger.info(f"Starting LoRA fine-tuning | batch={batch_id}")
        checkpoint_path = self.output_dir / batch_id
        checkpoint_path.mkdir(exist_ok=True)

        # ── QLoRA quantization config ────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config["load_in_4bit"],
            bnb_4bit_quant_type=self.config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, self.config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=True,
        )

        # ── Load base model ──────────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
        tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)

        # ── LoRA config ──────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["lora_target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Dataset ──────────────────────────────────────────────────────
        dataset = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(val_path)},
        )

        # ── Training arguments ───────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=str(checkpoint_path),
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_ratio=self.config["warmup_ratio"],
            evaluation_strategy="steps",
            eval_steps=self.config["eval_steps"],
            save_steps=self.config["save_steps"],
            logging_steps=10,
            fp16=True,
            optim="paged_adamw_32bit",
            report_to="none",
        )

        # ── SFT Trainer ──────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config["max_seq_length"],
            packing=True,
        )

        logger.info("Training started...")
        train_result = trainer.train()
        trainer.save_model(str(checkpoint_path / "final"))

        final_loss = train_result.training_loss
        logger.info(f"Training complete | loss={final_loss:.4f} | checkpoint={checkpoint_path}")

        result = {
            "batch_id": batch_id,
            "checkpoint_path": str(checkpoint_path / "final"),
            "training_loss": final_loss,
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["validation"]),
        }

        # Update batch status in store
        if store:
            batch = TrainingBatch(
                id=batch_id,
                record_ids=[],
                record_count=len(dataset["train"]) + len(dataset["validation"]),
                avg_quality_score=0,
                status="completed",
                model_checkpoint=str(checkpoint_path / "final"),
                training_loss=final_loss,
            )
            store.save_training_batch(batch)

        return result


class FineTuningPipeline:
    """
    Full fine-tuning pipeline orchestrator.
    Call `.run()` when enough approved data has been collected.
    """

    MIN_RECORDS_FOR_TRAINING = 50   # Minimum before attempting fine-tuning

    def __init__(
        self,
        store: Optional[FeedbackStore] = None,
        trainer_config: Optional[Dict[str, Any]] = None,
    ):
        self.store = store or FeedbackStore()
        self.dataset_builder = DatasetBuilder()
        self.trainer = LoRATrainer(config=trainer_config)

    def check_readiness(self, min_score: int = 3) -> Dict[str, Any]:
        """Check if there's enough data to start training."""
        stats = self.store.get_stats()
        ready_count = len(self.store.get_approved_unused(min_score=min_score))
        deps = self.trainer.check_dependencies()

        return {
            "ready_for_training": ready_count >= self.MIN_RECORDS_FOR_TRAINING,
            "approved_unused_records": ready_count,
            "min_required": self.MIN_RECORDS_FOR_TRAINING,
            "deps_installed": all(deps.values()),
            "missing_deps": [k for k, v in deps.items() if not v],
            "store_stats": stats,
        }

    def run(self, min_score: int = 3, dry_run: bool = False) -> Dict[str, Any]:
        """
        Full pipeline: fetch data → build dataset → train → update store.

        Args:
            min_score: Minimum quality score for training records
            dry_run: If True, prepare dataset but skip actual training
        """
        readiness = self.check_readiness(min_score)

        if not readiness["ready_for_training"]:
            logger.warning(
                f"Not enough data: {readiness['approved_unused_records']} / "
                f"{self.MIN_RECORDS_FOR_TRAINING} records ready"
            )
            return {"status": "insufficient_data", **readiness}

        if not readiness["deps_installed"] and not dry_run:
            logger.error(f"Missing deps: {readiness['missing_deps']}")
            return {"status": "missing_dependencies", **readiness}

        # Fetch records
        records = self.store.get_approved_unused(min_score=min_score)
        batch_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting training batch {batch_id} with {len(records)} records")

        # Build dataset
        dataset_paths = self.dataset_builder.build_from_records(records, min_score=min_score)

        if dry_run:
            return {
                "status": "dry_run_complete",
                "batch_id": batch_id,
                "records": len(records),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
            }

        # Train
        result = self.trainer.train(
            train_path=dataset_paths["train"],
            val_path=dataset_paths["val"],
            batch_id=batch_id,
            store=self.store,
        )

        # Mark records as used
        record_ids = [r.id for r in records]
        self.store.mark_used_in_training(record_ids, batch_id)

        return {"status": "completed", **result}
