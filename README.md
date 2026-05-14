# Architectural AI — Demo Project
**Ingoude Company** · AI-powered CAD drawing generation

## Architecture

```
User Prompt
    → Ollama (Prompt Enhancement) — 3 enhanced variants
    → Claude API (Action Plan + JSON AST) — normative knowledge base
    → ASTValidator (geometry checks)
    → Claude API (CAD Code / Python for Revit or AutoCAD)
    → DataCollector (Human-in-the-Loop feedback)
    → LoRA Fine-Tuning Pipeline (when 50+ approved records)
```

## Modules

| Module | Technology | Status |
|--------|-----------|--------|
| `src/prompt_engine/` | Ollama (llama3.2) | ✅ Ready |
| `src/cad_generator/` | Claude API | ✅ Ready |
| `src/knowledge_base/` | СНиП/SP norms | ✅ Ready |
| `src/fine_tuning/` | LoRA/QLoRA (PEFT+TRL) | 🔧 Prepared |
| `src/pipeline.py` | Full orchestration | ✅ Ready |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Start Ollama
ollama pull llama3.2
ollama serve

# Run demo
python demo.py
```

## Fine-Tuning Module

The fine-tuning pipeline (`src/fine_tuning/`) is fully implemented but **not yet connected** to the main generation flow.

### When to activate:
- Collect ≥ 50 approved feedback records
- Install training deps: `pip install transformers peft trl accelerate bitsandbytes datasets`
- Call: `pipeline.run_fine_tuning(dry_run=False)`

### Data flow:
```
FeedbackRecord (SQLite)
    → DatasetBuilder → train.jsonl / val.jsonl
    → LoRATrainer (QLoRA on Mistral-7B or Llama-3)
    → checkpoint saved to data/checkpoints/
    → records marked as used_in_training
```

### Human-in-the-Loop:
```python
pipeline = ArchitecturalPipeline(config)
result = pipeline.run("5-storey residential building, Almaty")

# Architect reviews → submit feedback
pipeline.submit_feedback(
    record_id=result.record_id,
    quality_score=4,
    corrected_ast={...},  # optional corrections
    notes="Wall thickness adjusted"
)
```

## CAD Output

Current target: **AutoCAD (ezdxf)**

To switch to Revit:
```python
config = PipelineConfig(cad_target="revit")
```

Generated Python code uses:
- **AutoCAD**: `ezdxf` library, layers WALLS/COLUMNS/SLABS/OPENINGS
- **Revit**: `pyRevit` / Revit API, Transaction-based element creation
