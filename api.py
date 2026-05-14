"""
HTTP API — wraps the pipeline for Docker/production use.
"""

import os
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

from src.pipeline import ArchitecturalPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pipeline = ArchitecturalPipeline(
    PipelineConfig(
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        cad_target=os.getenv("CAD_TARGET", "autocad_ezdxf"),
        auto_record=True,
    )
)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def json_response(handler, status: int, data: dict):
    try:
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", len(body))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        logger.warning("Client disconnected before response was sent")


def read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(length)) if length else {}


class Handler(BaseHTTPRequestHandler):
    timeout = 300

    def log_message(self, fmt, *args):
        logger.info(fmt % args)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            json_response(self, 200, {"status": "ok"})
        elif path == "/stats":
            json_response(self, 200, pipeline.data_collector.get_stats())
        elif path == "/readiness":
            json_response(self, 200, pipeline.check_training_readiness())
        elif path.startswith("/download/"):
            self._serve_file(path[len("/download/"):])
        else:
            json_response(self, 404, {"error": "not found"})

    def _serve_file(self, filename: str):
        import mimetypes
        # Strip directory traversal
        filename = os.path.basename(filename)
        filepath = os.path.join("outputs", filename)
        if not os.path.isfile(filepath):
            return json_response(self, 404, {"error": "file not found"})
        mime, _ = mimetypes.guess_type(filename)
        if mime is None:
            mime = "application/octet-stream"
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected during file download")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = read_body(self)

            if path == "/generate":
                prompt = body.get("prompt", "")
                variant = body.get("variant_index", 0)
                if not prompt:
                    return json_response(self, 400, {"error": "prompt required"})
                result = pipeline.run(prompt, variant_index=variant)
                json_response(self, 200, {
                    "record_id": result.record_id,
                    "enhanced_variants": result.enhanced_prompt.enhanced_variants,
                    "missing_params": result.enhanced_prompt.missing_params,
                    "selected_variant": result.enhanced_prompt.selected_variant,
                    "ast": result.ast,
                    "cad_code": result.cad_code,
                    "is_valid": result.is_valid,
                    "validation_errors": result.validation_errors,
                    "validation_warnings": result.validation_warnings,
                    "dxf_path": result.dxf_path,
                    "obj_path": result.obj_path,
                })

            elif path == "/feedback":
                record_id = body.get("record_id")
                score = body.get("quality_score")
                if not record_id or not score:
                    return json_response(self, 400, {"error": "record_id and quality_score required"})
                ok = pipeline.submit_feedback(
                    record_id=record_id,
                    quality_score=int(score),
                    corrected_ast=body.get("corrected_ast"),
                    notes=body.get("notes"),
                )
                json_response(self, 200, {"ok": ok, "record_id": record_id})

            elif path == "/fine-tune":
                dry = body.get("dry_run", True)
                result = pipeline.run_fine_tuning(dry_run=dry)
                json_response(self, 200, result)

            else:
                json_response(self, 404, {"error": "not found"})

        except Exception as e:
            logger.exception("Request error")
            json_response(self, 500, {"error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Threaded API server on port {port}")
    server.serve_forever()
