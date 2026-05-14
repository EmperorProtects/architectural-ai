"""
Demo script — runs the full Architectural AI pipeline.
Usage: python demo.py
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.pipeline import ArchitecturalPipeline, PipelineConfig
from src.fine_tuning import FineTuningPipeline

DIVIDER = "─" * 60


def print_section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def demo_prompt_enhancement():
    """Demo: Ollama-based prompt enhancement only."""
    print_section("DEMO 1: Prompt Enhancement (Ollama)")

    from src.prompt_engine import PromptEnhancer

    enhancer = PromptEnhancer(model="llama3.2")

    user_prompt = "Хочу жилой дом на 5 этажей в Алматы"
    print(f"\n📝 Original prompt:\n   {user_prompt}")

    try:
        result = enhancer.process(user_prompt)

        print(f"\n✨ Enhanced variants ({len(result.enhanced_variants)}):")
        for i, v in enumerate(result.enhanced_variants, 1):
            print(f"\n  [{i}] {v[:200]}{'...' if len(v) > 200 else ''}")

        print(f"\n⚠️  Missing params: {result.missing_params or 'none'}")
        print(f"🏗️  Detected building type: {result.params.building_type if result.params else 'unknown'}")

    except Exception as e:
        print(f"\n⚠️  Ollama not available ({e}). Using fallback enhancement.")
        variants = [
            f"[Structural] 5-storey residential building in Almaty, frame structure, B25 concrete",
            f"[Materials] 5-storey residential in Almaty, brick facade, metal window frames",
            f"[Functional] 5-storey residential Almaty, 2-3BR apartments, underground parking",
        ]
        for i, v in enumerate(variants, 1):
            print(f"  [{i}] {v}")


def demo_ast_generation():
    """Demo: Claude-based AST generation."""
    print_section("DEMO 2: JSON AST Generation (Claude)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Showing example AST instead.")
        example_ast = {
            "building": {"type": "residential", "floors": 5, "total_area": 2500.0, "structural_system": "frame"},
            "site": {"width": 20000, "length": 30000, "orientation": "south"},
            "elements": [
                {"id": "col_001", "type": "column", "geometry": {"start": [0,0,0], "end": [0,0,3000], "width": 400, "height": 400}, "material": "B25"},
                {"id": "wall_001", "type": "wall", "geometry": {"start": [0,0,0], "end": [6000,0,0], "width": 200, "height": 3000}, "material": "brick"},
                {"id": "win_001", "type": "opening", "geometry": {"start": [1500,0,900], "end": [3000,0,900], "width": 1500, "height": 1500}, "material": "glass", "properties": {"parent_wall": "wall_001"}},
            ],
            "floors": [{"level": 1, "elevation": 0, "height": 3000, "rooms": []}],
            "metadata": {"generated_by": "architectural-ai/claude", "warnings": []},
        }
        print("\n📐 Example AST:")
        print(json.dumps(example_ast, indent=2, ensure_ascii=False))

        # Validate it
        from src.cad_generator import ASTValidator
        validator = ASTValidator()
        result = validator.validate(example_ast)
        print(f"\n✅ Validation: {'PASSED' if result.is_valid else 'FAILED'}")
        if result.warnings:
            print(f"⚠️  Warnings: {result.warnings}")
        return

    from src.cad_generator import ASTGenerator, ASTValidator

    generator = ASTGenerator(api_key=api_key)
    validator = ASTValidator()

    prompt = (
        "5-storey residential building, 2500m², plot 20x30m, "
        "frame structure B25 concrete, brick facade, south orientation, Almaty Kazakhstan"
    )

    print(f"\n📝 Prompt: {prompt}")
    print("\n⏳ Generating AST via Claude...")

    result = generator.generate_with_plan(prompt, building_type="residential")

    print(f"\n📋 Action plan steps: {len(result['action_plan'].get('steps', []))}")
    ast = result["ast"]
    print(f"📐 AST elements: {len(ast.get('elements', []))}")
    print(f"🏢 Floors: {len(ast.get('floors', []))}")

    val = validator.validate(ast)
    print(f"\n✅ Validation: {'PASSED' if val.is_valid else 'FAILED'}")
    if val.errors:
        print(f"❌ Errors: {val.errors}")
    if val.warnings:
        print(f"⚠️  Warnings: {val.warnings}")

    # Save AST
    output_path = Path("data/demo_ast.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ast, f, indent=2, ensure_ascii=False)
    print(f"\n💾 AST saved to {output_path}")


def demo_feedback_store():
    """Demo: Fine-tuning feedback store."""
    print_section("DEMO 3: Fine-Tuning Feedback Store")

    from src.fine_tuning import DataCollector, FeedbackStore

    collector = DataCollector()

    # Simulate a generation record
    fake_ast = {
        "building": {"type": "commercial", "floors": 3, "total_area": 1200.0, "structural_system": "frame"},
        "site": {"width": 15000, "length": 20000, "orientation": "north"},
        "elements": [],
        "floors": [],
        "metadata": {"generated_by": "demo", "warnings": []},
    }

    record_id = collector.record_generation(
        original_prompt="Small office building 3 floors",
        enhanced_prompt="3-storey commercial office building, 1200m², frame structure, glass facade",
        building_type="commercial",
        generated_ast=fake_ast,
        validation_errors=[],
        validation_warnings=["No elements defined"],
    )
    print(f"\n📝 Recorded generation: {record_id}")

    # Simulate architect feedback
    collector.submit_architect_feedback(
        record_id=record_id,
        quality_score=4,
        corrected_ast={**fake_ast, "elements": [{"id": "col_001", "type": "column"}]},
        notes="Looks good, added missing column element",
    )
    print(f"✅ Feedback submitted: score=4/5")

    # Stats
    stats = collector.get_stats()
    print(f"\n📊 Feedback Store Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Fine-tuning readiness
    pipeline = FineTuningPipeline(store=collector.store)
    readiness = pipeline.check_readiness()
    print(f"\n🎯 Training readiness:")
    print(f"   Ready: {readiness['ready_for_training']}")
    print(f"   Records: {readiness['approved_unused_records']} / {readiness['min_required']}")
    print(f"   Deps installed: {readiness['deps_installed']}")
    if readiness.get("missing_deps"):
        print(f"   Missing: {readiness['missing_deps']}")


def demo_full_pipeline():
    """Demo: Full end-to-end pipeline."""
    print_section("DEMO 4: Full Pipeline (Ollama + Claude)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  Set ANTHROPIC_API_KEY to run the full pipeline.")
        return

    config = PipelineConfig(
        ollama_model="llama3.2",
        anthropic_api_key=api_key,
        cad_target="autocad_ezdxf",
        auto_record=True,
    )
    pipeline = ArchitecturalPipeline(config=config)

    user_prompt = "Двухэтажный коммерческий объект, кафе, 400 кв.м, участок 15x20м"
    print(f"\n📝 User prompt: {user_prompt}")
    print("\n⏳ Running full pipeline...")

    result = pipeline.run(user_prompt, variant_index=0)

    print(f"\n{result.summary()}")

    if result.record_id:
        print(f"\n💬 Submitting architect feedback (score=5)...")
        pipeline.submit_feedback(
            record_id=result.record_id,
            quality_score=5,
            notes="Auto-approved in demo",
        )

    print(f"\n🎯 Training readiness: {pipeline.check_training_readiness()['ready_for_training']}")


if __name__ == "__main__":
    print("\n🏗️  ARCHITECTURAL AI DEMO")
    print("=" * 60)
    print("Ingoude Company — AI CAD Generation Pipeline")
    print("=" * 60)

    demo_prompt_enhancement()
    demo_ast_generation()
    demo_feedback_store()
    demo_full_pipeline()

    print(f"\n{DIVIDER}")
    print("  Demo complete.")
    print(DIVIDER)
