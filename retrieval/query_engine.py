"""
retrieval/query_engine.py
--------------------------
Orchestrates the full query processing pipeline for the
Contract Intelligence System.
 
Responsibilities:
- Accept a user query and optional session memory.
- Retrieve relevant contract chunks via SemanticRetriever.
- Generate a grounded answer via AnswerGenerator.
- Update short-term session memory with the latest turn.
- Return a complete QueryResponse with answer and sources.
 
This is the single entry point for the query tab in the UI.
 
Usage:
    from retrieval.query_engine import query_engine
 
    response = query_engine.run(
        query="What are the payment terms?",
        memory=st.session_state.memory,
    )
    print(response.answer)
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
 
from config.settings import settings
from llm.answer_generator import AnswerGenerator, AnswerResult, answer_generator
from retrieval.semantic_retriever import RetrievedContext, SemanticRetriever, semantic_retriever
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Query response dataclass
# ---------------------------------------------------------------------------
@dataclass
class QueryResponse:
    """
    The complete response from a single query processing run.
 
    Attributes:
        query:          The original user query.
        answer:         The generated answer text.
        sources:        List of source chunk texts used for the answer.
        source_files:   Distinct source filenames referenced.
        is_out_of_scope: Whether query was flagged as out of scope.
        is_grounded:    Whether the answer is grounded in the context.
        retrieval:      The full RetrievedContext object (for debugging).
    """
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    is_out_of_scope: bool = False
    is_grounded: bool = True
    retrieval: RetrievedContext | None = None
 
    def __repr__(self) -> str:
        return (
            f"QueryResponse("
            f"query='{self.query[:50]}', "
            f"grounded={self.is_grounded}, "
            f"sources={len(self.sources)})"
        )
 
 
# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------
class QueryEngine:
    """
    End-to-end query processor combining retrieval and answer generation.
 
    Wires together SemanticRetriever and AnswerGenerator into a single
    callable that the UI layer can use directly.
 
    Also integrates with short-term session memory to maintain
    conversational continuity across follow-up questions.
    """
 
    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever or semantic_retriever
        self._generator = generator or answer_generator
        logger.info("QueryEngine initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        memory=None,
        source_file: str | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """
        Run the full query pipeline: retrieve → generate → respond.
 
        Args:
            query:       The user's natural language question.
            memory:      Optional SessionMemory object for context enrichment.
            source_file: Restrict retrieval to a specific document.
            top_k:       Number of chunks to retrieve.
 
        Returns:
            QueryResponse with answer, sources, and metadata.
        """
        if not query or not query.strip():
            return QueryResponse(
                query=query,
                answer="Please enter a question.",
                is_grounded=False,
            )
 
        logger.info("QueryEngine.run | query='%s'", query[:80])
 
        # Step 1 — Enrich query with memory context if available
        enriched_query = self._enrich_with_memory(query, memory)
 
        # Step 2 — Retrieve relevant chunks
        try:
            retrieval = self._retriever.retrieve(
                query=enriched_query,
                top_k=top_k,
                source_file=source_file,
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return QueryResponse(
                query=query,
                answer="Retrieval failed. Please try again.",
                is_grounded=False,
            )
 
        # Step 3 — Generate answer
        try:
            result: AnswerResult = self._generator.generate(
                query=query,
                context=retrieval.combined_text,
            )
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return QueryResponse(
                query=query,
                answer="Answer generation failed. Please try again.",
                is_grounded=False,
            )
 
        # Step 4 — Update session memory
        if memory is not None:
            try:
                memory.add(
                    query=query,
                    answer=result.answer,
                    context=retrieval.combined_text,
                )
            except Exception as exc:
                logger.warning("Memory update failed (non-critical): %s", exc)
 
        # Step 5 — Build response
        sources = [chunk.text for chunk in retrieval.chunks]
        source_files = list({chunk.source_file for chunk in retrieval.chunks})
 
        response = QueryResponse(
            query=query,
            answer=result.answer,
            sources=sources,
            source_files=source_files,
            is_out_of_scope=result.is_out_of_scope,
            is_grounded=result.is_grounded,
            retrieval=retrieval,
        )
 
        logger.info(
            "QueryEngine complete | grounded=%s | sources=%d",
            response.is_grounded,
            len(response.sources),
        )
 
        return response
 
    def summarise(
        self,
        context: str,
        memory=None,
    ) -> QueryResponse:
        """
        Generate a structured summary of the contract.
 
        Args:
            context: Full or combined contract text to summarise.
            memory:  Optional SessionMemory object.
 
        Returns:
            QueryResponse containing the structured summary.
        """
        logger.info("QueryEngine.summarise | chars=%d", len(context))
 
        try:
            result = self._generator.summarise(context)
        except Exception as exc:
            logger.error("Summarisation failed: %s", exc)
            return QueryResponse(
                query="Summarise the contract.",
                answer="Summarisation failed. Please try again.",
                is_grounded=False,
            )
 
        if memory is not None:
            try:
                memory.add(
                    query="Summarise the contract.",
                    answer=result.answer,
                    context=context,
                )
            except Exception as exc:
                logger.warning("Memory update failed (non-critical): %s", exc)
 
        return QueryResponse(
            query="Summarise the contract.",
            answer=result.answer,
            sources=[],
            source_files=[],
            is_grounded=True,
        )
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _enrich_with_memory(
        self,
        query: str,
        memory,
    ) -> str:
        """
        Optionally prepend recent conversation history to the query
        for better contextual retrieval on follow-up questions.
 
        Args:
            query:  Current user query.
            memory: SessionMemory object or None.
 
        Returns:
            Enriched query string, or original query if no memory.
        """
        if memory is None:
            return query
 
        recent = memory.get_recent_context()
        if not recent:
            return query
 
        enriched = f"{recent}\n\nCurrent question: {query}"
        logger.debug("Query enriched with memory context.")
        return enriched
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
query_engine = QueryEngine()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python retrieval/query_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== QueryEngine Smoke Test ===\n")
 
    engine = QueryEngine()
 
    test_queries = [
        "Who are the parties in the agreement?",
        "What are the payment terms?",
        "What is the termination notice period?",
        "What is the weather today?",
    ]
 
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            response = engine.run(query)
            print(f"  Out of scope : {response.is_out_of_scope}")
            print(f"  Grounded     : {response.is_grounded}")
            print(f"  Sources      : {len(response.sources)}")
            print(f"  Source files : {response.source_files}")
            print(f"  Answer       : {response.answer[:250]}")
            print(f"  PASSED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")
