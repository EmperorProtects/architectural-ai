"""
Main Pipeline — orchestrates the full workflow:
Prompt → Enhancement (Ollama) → AST (Claude) → Validation → CAD Code → Feedback
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import uuid
from pathlib import Path
from .prompt_engine import PromptEnhancer, EnhancedPrompt
from .cad_generator import ASTGenerator, ASTValidator, CADTranslator, build_dxf, build_obj
from .fine_tuning import DataCollector, FineTuningPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    claude_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: Optional[str] = None
    cad_target: str = "autocad_ezdxf"
    auto_record: bool = True   # Automatically save to feedback store


@dataclass
class PipelineResult:
    enhanced_prompt: EnhancedPrompt
    action_plan: Dict[str, Any]
    ast: Dict[str, Any]
    validation_errors: list
    validation_warnings: list
    cad_code: str
    record_id: Optional[str]
    is_valid: bool
    dxf_path: Optional[str] = None
    obj_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"✓ Enhanced variants: {len(self.enhanced_prompt.enhanced_variants)}",
            f"✓ Missing params: {self.enhanced_prompt.missing_params or 'none'}",
            f"✓ AST elements: {len(self.ast.get('elements', []))}",
            f"✓ Floors: {len(self.ast.get('floors', []))}",
            f"{'✓' if self.is_valid else '✗'} Validation: "
            f"{'passed' if self.is_valid else str(self.validation_errors)}",
            f"✓ CAD code: {len(self.cad_code.splitlines())} lines",
        ]
        if self.dxf_path:
            lines.append(f"✓ DXF (2D): {self.dxf_path}")
        if self.obj_path:
            lines.append(f"✓ OBJ (3D): {self.obj_path}")
        if self.record_id:
            lines.append(f"✓ Recorded: {self.record_id}")
        return "\n".join(lines)


class ArchitecturalPipeline:
    """
    Full pipeline:
    User Prompt
        → Ollama: Prompt Enhancement (3 variants)
        → Ollama: Parameter Validation
        → Claude: Action Plan
        → Claude: JSON AST Generation
        → ASTValidator: Geometry check
        → Claude: CAD Code Translation
        → DataCollector: Record for fine-tuning
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.enhancer = PromptEnhancer(
            model=self.config.ollama_model,
            base_url=self.config.ollama_url,
        )
        self.ast_generator = ASTGenerator(
            api_key=self.config.anthropic_api_key,
            model=self.config.claude_model,
        )
        self.ast_validator = ASTValidator()
        self.cad_translator = CADTranslator(
            api_key=self.config.anthropic_api_key,
            model=self.config.claude_model,
        )
        self.data_collector = DataCollector()
        self.fine_tuning = FineTuningPipeline(store=self.data_collector.store)

    def run(
        self,
        user_prompt: str,
        variant_index: int = 0,  # kept for backwards-compatible API; ignored
        generate_cad: bool = True,
    ) -> PipelineResult:
        """Execute the full pipeline for a user prompt."""
        logger.info(f"Pipeline start: '{user_prompt[:60]}...'")

        # ── Step 1: Prompt Enhancement (Ollama) ─────────────────────────
        logger.info("[1/5] Enhancing prompt via Ollama...")
        enhanced = self.enhancer.process(user_prompt)
        final_prompt = enhanced.selected_variant or user_prompt
        building_type = "residential"

        # ── Step 2: AST Generation (Claude) ─────────────────────────────
        logger.info("[2/5] Generating JSON AST via Claude...")
        generation = self.ast_generator.generate_with_plan(final_prompt, building_type)
        action_plan = generation["action_plan"]
        ast = generation["ast"]

        # ── Step 3: AST Validation ───────────────────────────────────────
        logger.info("[3/5] Validating AST geometry...")
        validation = self.ast_validator.validate(ast)
        ast = validation.corrected_ast  # Use corrected version

        # ── Step 4: CAD Code Generation (Claude) ─────────────────────────
        cad_code = ""
        if generate_cad and validation.is_valid:
            logger.info(f"[4/5] Translating AST to {self.config.cad_target} code...")
            cad_code = self.cad_translator.translate(ast, target=self.config.cad_target)
        elif not validation.is_valid:
            logger.warning(f"[4/5] Skipping CAD generation — AST invalid: {validation.errors}")

        # ── Step 5: Generate output files (DXF 2D + OBJ 3D) ─────────────
        run_id = str(uuid.uuid4())[:8]
        dxf_path: Optional[str] = None
        obj_path: Optional[str] = None
        if validation.is_valid:
            try:
                logger.info("[5a] Building DXF (2D floor plan)...")
                dxf_path = build_dxf(ast, output_path=f"outputs/{run_id}_2d.dxf")
            except Exception as e:
                logger.warning(f"DXF build failed: {e}")
            try:
                logger.info("[5b] Building OBJ (3D model)...")
                obj_path = build_obj(ast, output_path=f"outputs/{run_id}_3d.obj")
            except Exception as e:
                logger.warning(f"OBJ build failed: {e}")

        # ── Step 6: Record for Fine-Tuning ───────────────────────────────
        record_id = None
        if self.config.auto_record:
            logger.info("[6/6] Recording to feedback store...")
            record_id = self.data_collector.record_generation(
                original_prompt=user_prompt,
                enhanced_prompt=final_prompt,
                building_type=building_type,
                generated_ast=ast,
                action_plan=action_plan,
                generated_cad_code=cad_code,
                cad_target=self.config.cad_target,
                validation_errors=validation.errors,
                validation_warnings=validation.warnings,
            )

        result = PipelineResult(
            enhanced_prompt=enhanced,
            action_plan=action_plan,
            ast=ast,
            validation_errors=validation.errors,
            validation_warnings=validation.warnings,
            cad_code=cad_code,
            record_id=record_id,
            is_valid=validation.is_valid,
            dxf_path=dxf_path,
            obj_path=obj_path,
        )

        logger.info(f"Pipeline complete:\n{result.summary()}")
        return result

    def submit_feedback(
        self,
        record_id: str,
        quality_score: int,
        corrected_ast: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Submit architect feedback for a completed generation."""
        return self.data_collector.submit_architect_feedback(
            record_id=record_id,
            quality_score=quality_score,
            corrected_ast=corrected_ast,
            notes=notes,
            approve=True,
        )

    def check_training_readiness(self) -> dict:
        """Check if enough data has been collected for fine-tuning."""
        return self.fine_tuning.check_readiness()

    def run_fine_tuning(self, dry_run: bool = True) -> dict:
        """Trigger fine-tuning pipeline."""
        return self.fine_tuning.run(dry_run=dry_run)
