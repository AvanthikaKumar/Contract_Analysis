"""
ingestion/embedder.py
----------------------
Generates vector embeddings for contract chunks and queries
using the Azure OpenAI embedding model.
 
Responsibilities:
- Embed a single text string (used for query embedding at retrieval time).
- Embed a list of Chunk objects in batches (used during ingestion).
- Attach embedding vectors directly to Chunk objects.
- Respect Azure OpenAI batch size limits.
 
Usage:
    from ingestion.embedder import embedder
    from ingestion.chunker import Chunk
 
    # Embed a query
    vector = embedder.embed_query("What are the payment terms?")
 
    # Embed chunks during ingestion
    embedded_chunks = embedder.embed_chunks(chunks)
"""
 
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from ingestion.chunker import Chunk
from llm.azure_openai_client import openai_client
 
logger = logging.getLogger(__name__)
 
# Azure OpenAI allows up to 2048 inputs per batch request
# We use a conservative limit to stay well within limits
_MAX_BATCH_SIZE = 16
 
 
# ---------------------------------------------------------------------------
# Embedded chunk
# ---------------------------------------------------------------------------
@dataclass
class EmbeddedChunk:
    """
    A Chunk that has been enriched with its embedding vector.
 
    Attributes:
        chunk:     The original Chunk object.
        embedding: The vector embedding as a list of floats.
    """
    chunk: Chunk
    embedding: list[float] = field(default_factory=list)
 
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
 
    @property
    def text(self) -> str:
        return self.chunk.text
 
    @property
    def metadata(self) -> dict:
        return self.chunk.metadata
 
    def __repr__(self) -> str:
        return (
            f"EmbeddedChunk(id='{self.chunk_id}', "
            f"dims={len(self.embedding)})"
        )
 
 
# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class Embedder:
    """
    Generates embeddings for chunks and queries via Azure OpenAI.
 
    Batches chunk embedding requests to reduce API call overhead.
    Uses the same openai_client singleton from Phase 2.
    """
 
    def __init__(self) -> None:
        logger.info("Embedder initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for semantic search.
 
        Args:
            query: The natural language query to embed.
 
        Returns:
            Embedding vector as a list of floats.
 
        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
 
        logger.info("Embedding query | length=%d", len(query))
        vector = openai_client.get_embedding(query)
        logger.info("Query embedded | dims=%d", len(vector))
        return vector
 
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunk objects in batches.
 
        Args:
            chunks: List of Chunk objects from the Chunker.
 
        Returns:
            List of EmbeddedChunk objects in the same order as input.
 
        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot embed an empty list of chunks.")
 
        logger.info(
            "Starting chunk embedding | total_chunks=%d | batch_size=%d",
            len(chunks),
            _MAX_BATCH_SIZE,
        )
 
        embedded: list[EmbeddedChunk] = []
        batches = self._make_batches(chunks, _MAX_BATCH_SIZE)
 
        for batch_num, batch in enumerate(batches, start=1):
            logger.info(
                "Embedding batch %d/%d | chunks=%d",
                batch_num,
                len(batches),
                len(batch),
            )
 
            texts = [chunk.text for chunk in batch]
            vectors = openai_client.get_embeddings_batch(texts)
 
            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(chunk=chunk, embedding=vector))
 
        logger.info(
            "All chunks embedded | total=%d | dims=%d",
            len(embedded),
            len(embedded[0].embedding) if embedded else 0,
        )
 
        return embedded
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_batches(
        items: list[Chunk],
        batch_size: int,
    ) -> list[list[Chunk]]:
        """Split a list into sub-lists of at most batch_size items."""
        return [
            items[i : i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
embedder = Embedder()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python ingestion/embedder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ingestion.chunker import Chunk
 
    print("\n=== Embedder Smoke Test ===\n")
 
    # Build sample chunks without needing a real PDF
    sample_chunks = [
        Chunk(
            chunk_id="test_chunk_0000",
            text="The agreement commences on 1 January 2025 and expires on 31 December 2025.",
            index=0,
            char_start=0,
            char_end=75,
            source_file="test_contract.pdf",
        ),
        Chunk(
            chunk_id="test_chunk_0001",
            text="Payment is due within 30 days of invoice. Late payments attract 1.5% monthly interest.",
            index=1,
            char_start=75,
            char_end=160,
            source_file="test_contract.pdf",
        ),
    ]
 
    emb = Embedder()
 
    print("1. Testing embed_query()...")
    try:
        vector = emb.embed_query("What are the payment terms?")
        print(f"   Dimensions : {len(vector)}")
        print(f"   First 5    : {[round(v, 4) for v in vector[:5]]}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("2. Testing embed_chunks()...")
    try:
        embedded = emb.embed_chunks(sample_chunks)
        for ec in embedded:
            print(f"   {ec.chunk_id} | dims={len(ec.embedding)}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")