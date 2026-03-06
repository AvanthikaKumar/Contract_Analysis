"""
core/prompt_manager.py
-----------------------
Prompt governance layer for the Contract Intelligence System.
 
- Loads prompt templates from .md files in the /prompts directory.
- Injects runtime variables using {placeholder} syntax.
- Caches loaded prompts to avoid repeated disk reads.
- Ensures no prompts are ever hardcoded in application logic.
 
Usage:
    from core.prompt_manager import prompt_manager
 
    prompt = prompt_manager.load("answer_prompt", variables={
        "context": retrieved_chunks,
        "question": user_query,
    })
"""
 
import logging
import sys
from pathlib import Path
from typing import Optional
 
# ---------------------------------------------------------------------------
# Guarantee project root is on sys.path when this file is run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from config.settings import settings  # noqa: E402 — must come after path fix
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class PromptNotFoundError(FileNotFoundError):
    """Raised when a requested prompt file does not exist."""
 
 
class PromptRenderError(KeyError):
    """Raised when a required template variable is missing."""
 
 
# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------
class PromptManager:
    """
    Loads, caches, and renders prompt templates from Markdown files.
 
    All prompts live in the /prompts directory as .md files.
    Variables are injected using Python's str.format_map() so templates
    use standard {variable_name} placeholders.
    """
 
    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self.prompts_dir: Path = prompts_dir or settings.app.prompts_dir
        self._cache: dict[str, str] = {}
        logger.info("PromptManager initialised. Prompts dir: %s", self.prompts_dir)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def load(
        self,
        prompt_name: str,
        variables: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Load a prompt by name, optionally injecting template variables.
 
        Args:
            prompt_name: Filename without extension e.g. "answer_prompt"
            variables:   Dict of {placeholder: value} to inject.
 
        Returns:
            Rendered prompt string.
 
        Raises:
            PromptNotFoundError: If the .md file does not exist.
            PromptRenderError:   If a required placeholder is missing.
        """
        raw_template = self._get_raw(prompt_name)
        if not variables:
            return raw_template
        return self._render(prompt_name, raw_template, variables)
 
    def reload(self, prompt_name: str) -> None:
        """Force-reload a prompt from disk, clearing its cache entry."""
        key = self._cache_key(prompt_name)
        if key in self._cache:
            del self._cache[key]
            logger.info("Cache cleared for prompt: '%s'", prompt_name)
        self._get_raw(prompt_name)
 
    def reload_all(self) -> None:
        """Clear entire prompt cache."""
        self._cache.clear()
        logger.info("Full prompt cache cleared.")
 
    def list_available(self) -> list[str]:
        """Return names of all available prompts (without .md extension)."""
        if not self.prompts_dir.exists():
            logger.warning("Prompts directory not found: %s", self.prompts_dir)
            return []
        return sorted(p.stem for p in self.prompts_dir.glob("*.md"))
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _cache_key(self, prompt_name: str) -> str:
        return prompt_name.lower().strip()
 
    def _get_raw(self, prompt_name: str) -> str:
        key = self._cache_key(prompt_name)
        if key in self._cache:
            logger.debug("Cache hit for prompt: '%s'", prompt_name)
            return self._cache[key]
 
        file_path = self.prompts_dir / f"{prompt_name}.md"
        if not file_path.exists():
            raise PromptNotFoundError(
                f"Prompt file not found: '{file_path}'\n"
                f"Available prompts: {self.list_available()}"
            )
 
        raw = file_path.read_text(encoding="utf-8").strip()
        self._cache[key] = raw
        logger.info("Loaded prompt '%s' (%d chars).", prompt_name, len(raw))
        return raw
 
    def _render(
        self,
        prompt_name: str,
        template: str,
        variables: dict[str, str],
    ) -> str:
        class _StrictDefault(dict):
            def __missing__(self, key: str) -> str:
                raise PromptRenderError(
                    f"Prompt '{prompt_name}' requires variable '{{{key}}}' "
                    f"but it was not provided. Supplied keys: {list(variables.keys())}"
                )
 
        try:
            rendered = template.format_map(_StrictDefault(variables))
            logger.debug("Rendered prompt '%s' with vars: %s", prompt_name, list(variables.keys()))
            return rendered
        except PromptRenderError:
            raise
        except Exception as exc:
            raise PromptRenderError(f"Failed to render prompt '{prompt_name}': {exc}") from exc
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
prompt_manager = PromptManager()
 
 
# ---------------------------------------------------------------------------
# Smoke test — run directly: python core/prompt_manager.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pm = PromptManager()
 
    print("\n=== Available Prompts ===")
    available = pm.list_available()
    if not available:
        print("  No prompt files found. Check prompts/ directory.")
    for name in available:
        print(f"  - {name}")
 
    print("\n=== Loading answer_prompt with test variables ===")
    try:
        rendered = pm.load(
            "answer_prompt",
            variables={
                "context": "The agreement commences on 1 January 2025 and expires on 31 December 2025.",
                "question": "What is the contract start date?",
            },
        )
        print(rendered)
    except PromptNotFoundError as exc:
        print(f"[PromptNotFoundError] {exc}")
 
    print("\n=== Testing missing variable error handling ===")
    try:
        pm.load("answer_prompt", variables={"context": "some context"})
    except PromptRenderError as exc:
        print(f"[PromptRenderError caught correctly] {exc}")
 
    print("\nPromptManager smoke test complete.")
 
