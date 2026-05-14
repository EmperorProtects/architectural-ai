import logging
import httpx
from .models import EnhancedPrompt

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

ENHANCE_SYSTEM_PROMPT = """You are an expert architectural AI assistant specializing in CAD drawing generation.

Your task: rewrite the user's architectural prompt as a single precise, technical prompt ready for CAD generation.

Rules:
- Add specific dimensions where missing (use standard building norms)
- Specify materials with technical grade where applicable
- Include structural system references (frame, bearing walls, etc.)
- Reference building codes (СНиП, ГОСТ for CIS region)
- Be specific about room types, circulation, MEP zones
- Output plain text only — no JSON, no markdown, no preamble. Return only the rewritten prompt."""


class OllamaPromptEnhancer:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def _call_ollama(self, prompt: str, system: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def enhance(self, user_prompt: str) -> str:
        """Generate a single enhanced version of the user prompt via Ollama."""
        logger.info(f"Enhancing prompt: {user_prompt[:80]}...")
        try:
            raw = self._call_ollama(
                prompt=f"Enhance this architectural prompt:\n\n{user_prompt}",
                system=ENHANCE_SYSTEM_PROMPT,
            )
            text = (raw or "").strip().strip("`").strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"Ollama enhancement failed: {e}. Using original prompt.")
        return user_prompt

    def process(self, user_prompt: str) -> EnhancedPrompt:
        enhanced = self.enhance(user_prompt)
        return EnhancedPrompt(
            original=user_prompt,
            enhanced_variants=[enhanced],
            missing_params=[],
            params=None,
            selected_variant=enhanced,
        )


class PromptEnhancer(OllamaPromptEnhancer):
    """Public alias."""
    pass
