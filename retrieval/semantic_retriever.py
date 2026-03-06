"""
retrieval/semantic_retriever.py
--------------------------------
Performs semantic vector search against Azure AI Search
to retrieve the most relevant contract chunks for a given query.
 
Responsibilities:
- Accept a natural language query string.
- Convert query to an embedding vector.
- Search Azure AI Search for top-K relevant chunks.
- Return clean RetrievedContext objects ready for the LLM.
 
Usage:
    from retrieval.semantic_retriever import semantic_retriever
 
    context = semantic_retriever.retrieve("What are the payment terms?")
"""
 
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from config.settings import settings
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store, SearchResult
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Retrieved context dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievedContext:
    """
    The result of a semantic retrieval operation.
 
    Attributes:
        query:          The original user query.
        chunks:         List of SearchResult objects ordered by relevance.
        combined_text:  All chunk texts joined for direct LLM injection.
        top_k:          Number of results requested.
    """
    query: str
    chunks: list[SearchResult]
    combined_text: str
    top_k: int
 
    def __repr__(self) -> str:
        return (
            f"RetrievedContext("
            f"query='{self.query[:50]}...', "
            f"chunks={len(self.chunks)}, "
            f"chars={len(self.combined_text)})"
        )
 
 
# ---------------------------------------------------------------------------
# Semantic retriever
# ---------------------------------------------------------------------------
class SemanticRetriever:
    """
    Orchestrates query embedding and vector search to retrieve
    relevant contract chunks from Azure AI Search.
    """
 
    def __init__(self) -> None:
        logger.info("SemanticRetriever initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> RetrievedContext:
        """
        Retrieve the most relevant contract chunks for a query.
 
        Args:
            query:       Natural language question from the user.
            top_k:       Number of chunks to retrieve. Defaults to
                         settings.app.retrieval_top_k.
            source_file: Optional filter to search within one document only.
 
        Returns:
            RetrievedContext with ranked chunks and combined text.
 
        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
 
        k = top_k or settings.app.retrieval_top_k
 
        logger.info(
            "Retrieving context | query='%s' | top_k=%d | filter=%s",
            query[:80],
            k,
            source_file or "none",
        )
 
        # Step 1 — Embed the query
        query_vector = embedder.embed_query(query)
 
        # Step 2 — Vector search
        results = vector_store.search(
            query_vector=query_vector,
            top_k=k,
            source_file=source_file,
        )
 
        # Step 3 — Build combined context text for LLM injection
        combined_text = self._build_combined_text(results)
 
        logger.info(
            "Retrieval complete | chunks=%d | context_chars=%d",
            len(results),
            len(combined_text),
        )
 
        return RetrievedContext(
            query=query,
            chunks=results,
            combined_text=combined_text,
            top_k=k,
        )
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_combined_text(self, results: list[SearchResult]) -> str:
        """
        Join retrieved chunk texts into a single context block.
 
        Each chunk is separated by a clear delimiter so the LLM
        can distinguish between different parts of the contract.
 
        Args:
            results: Ordered list of SearchResult objects.
 
        Returns:
            Combined context string for LLM injection.
        """
        if not results:
            return ""
 
        parts = []
        for i, result in enumerate(results, start=1):
            parts.append(
                f"[Excerpt {i} — Source: {result.source_file}]\n"
                f"{result.text}"
            )
 
        return "\n\n---\n\n".join(parts)
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
semantic_retriever = SemanticRetriever()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python retrieval/semantic_retriever.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== SemanticRetriever Smoke Test ===\n")
 
    retriever = SemanticRetriever()
 
    test_queries = [
        "What are the payment terms?",
        "Who are the parties in the agreement?",
        "What are the termination conditions?",
    ]
 
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            context = retriever.retrieve(query, top_k=3)
            print(f"  Chunks retrieved : {len(context.chunks)}")
            print(f"  Context chars    : {len(context.combined_text)}")
            if context.chunks:
                print(f"  Top result score : {context.chunks[0].score:.4f}")
                print(f"  Top result source: {context.chunks[0].source_file}")
                print(f"  Preview          : {context.chunks[0].text[:150]}...")
            print(f"  PASSED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")
 
