"""
llm/azure_openai_client.py
--------------------------
Central Azure OpenAI client for the Contract Intelligence System.
 
Responsibilities:
- Provides a single initialised AzureOpenAI client reused across the app.
- Exposes two clean methods:
    1. get_chat_completion()  — for LLM answer generation
    2. get_embedding()        — for vectorising text (queries + chunks)
- Handles retries with exponential backoff for transient Azure errors.
- Logs token usage for observability.
 
Usage:
    from llm.azure_openai_client import openai_client
 
    response = openai_client.get_chat_completion(messages=[...])
    vector   = openai_client.get_embedding("some text")
"""
 
import logging
import sys
from pathlib import Path
from typing import Optional
 
# ---------------------------------------------------------------------------
# Path fix — guarantees imports work when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from openai import AzureOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Retry policy — applied to all Azure OpenAI calls
# ---------------------------------------------------------------------------
_RETRY_POLICY = dict(
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
 
 
# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------
class AzureOpenAIClient:
    """
    Thin, retry-enabled wrapper around the AzureOpenAI SDK client.
 
    Attributes:
        _client: The underlying AzureOpenAI SDK instance.
        _chat_deployment: Azure deployment name for chat completions.
        _embedding_deployment: Azure deployment name for embeddings.
    """
 
    def __init__(self) -> None:
        cfg = settings.azure_openai
        self._client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
        )
        self._chat_deployment = cfg.chat_deployment
        self._embedding_deployment = cfg.embedding_deployment
        logger.info(
            "AzureOpenAIClient initialised | chat=%s | embedding=%s",
            self._chat_deployment,
            self._embedding_deployment,
        )
 
    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------
    @retry(**_RETRY_POLICY)
    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1500,
        deployment: Optional[str] = None,
    ) -> str:
        """
        Send a list of messages to Azure OpenAI and return the reply text.
 
        Args:
            messages:    List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Maximum tokens in the completion.
            deployment:  Override the default chat deployment if needed.
 
        Returns:
            Assistant reply as a plain string.
 
        Raises:
            APIError | RateLimitError | APITimeoutError: After retries exhausted.
        """
        target_deployment = deployment or self._chat_deployment
 
        logger.debug(
            "Chat completion request | deployment=%s | messages=%d | temp=%.1f",
            target_deployment,
            len(messages),
            temperature,
        )
 
        response = self._client.chat.completions.create(
            model=target_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
 
        reply = response.choices[0].message.content or ""
 
        # Log token usage for observability
        usage = response.usage
        if usage:
            logger.info(
                "Chat completion done | prompt_tokens=%d | completion_tokens=%d | total=%d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
 
        return reply.strip()
 
    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    @retry(**_RETRY_POLICY)
    def get_embedding(
        self,
        text: str,
        deployment: Optional[str] = None,
    ) -> list[float]:
        """
        Generate a vector embedding for the given text.
 
        Args:
            text:       The input string to embed.
            deployment: Override the default embedding deployment if needed.
 
        Returns:
            Embedding as a list of floats.
 
        Raises:
            ValueError: If text is empty.
            APIError | RateLimitError | APITimeoutError: After retries exhausted.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
 
        target_deployment = deployment or self._embedding_deployment
 
        # Normalise whitespace — improves embedding consistency
        normalised = " ".join(text.split())
 
        logger.debug(
            "Embedding request | deployment=%s | text_length=%d",
            target_deployment,
            len(normalised),
        )
 
        response = self._client.embeddings.create(
            model=target_deployment,
            input=normalised,
        )
 
        vector = response.data[0].embedding
 
        logger.info(
            "Embedding done | deployment=%s | dimensions=%d",
            target_deployment,
            len(vector),
        )
 
        return vector
 
    # ------------------------------------------------------------------
    # Batch embeddings
    # ------------------------------------------------------------------
    @retry(**_RETRY_POLICY)
    def get_embeddings_batch(
        self,
        texts: list[str],
        deployment: Optional[str] = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts in a single API call.
 
        Args:
            texts:      List of strings to embed.
            deployment: Override the default embedding deployment.
 
        Returns:
            List of embedding vectors, in the same order as input texts.
 
        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")
 
        target_deployment = deployment or self._embedding_deployment
        normalised = [" ".join(t.split()) for t in texts]
 
        logger.debug(
            "Batch embedding request | deployment=%s | count=%d",
            target_deployment,
            len(normalised),
        )
 
        response = self._client.embeddings.create(
            model=target_deployment,
            input=normalised,
        )
 
        # SDK returns results sorted by index — preserve order explicitly
        sorted_data = sorted(response.data, key=lambda d: d.index)
        vectors = [d.embedding for d in sorted_data]
 
        logger.info(
            "Batch embedding done | count=%d | dimensions=%d",
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
 
        return vectors
 
 
# ---------------------------------------------------------------------------
# Module-level singleton — import and reuse everywhere
# ---------------------------------------------------------------------------
openai_client = AzureOpenAIClient()
 
 
# ---------------------------------------------------------------------------
# Smoke test — run directly: python llm/azure_openai_client.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== AzureOpenAIClient Smoke Test ===\n")
 
    client = AzureOpenAIClient()
 
    # Test embedding
    print("1. Testing get_embedding()...")
    try:
        vector = client.get_embedding("This is a test contract clause.")
        print(f"   Embedding dimensions : {len(vector)}")
        print(f"   First 5 values       : {vector[:5]}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    # Test batch embedding
    print("2. Testing get_embeddings_batch()...")
    try:
        vectors = client.get_embeddings_batch([
            "The agreement commences on 1 January 2025.",
            "Payment is due within 30 days of invoice.",
        ])
        print(f"   Batch count          : {len(vectors)}")
        print(f"   Each dimension       : {len(vectors[0])}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    # Test chat completion
    print("3. Testing get_chat_completion()...")
    try:
        reply = client.get_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with exactly: Azure OpenAI connection successful."},
            ]
        )
        print(f"   Response : {reply}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")
 
