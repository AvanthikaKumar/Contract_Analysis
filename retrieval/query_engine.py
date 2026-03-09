"""
retrieval/query_engine.py
--------------------------
Orchestrates the full query processing pipeline.
"""
 
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from config.settings import settings
from llm.answer_generator import AnswerGenerator, AnswerResult, answer_generator
from retrieval.semantic_retriever import RetrievedContext, SemanticRetriever, semantic_retriever
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class QueryResponse:
    query: str
    answer: str
    sources: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    is_out_of_scope: bool = False
    is_grounded: bool = True
    retrieval: RetrievedContext | None = None
 
 
class QueryEngine:
 
    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever or semantic_retriever
        self._generator = generator or answer_generator
        logger.info("QueryEngine initialised.")
 
    def run(
        self,
        query: str,
        memory=None,
        source_file: str | None = None,
        top_k: int | None = None,
        known_files: list[str] | None = None,
    ) -> QueryResponse:
        if not query or not query.strip():
            return QueryResponse(query=query, answer="Please enter a question.", is_grounded=False)
 
        logger.info("QueryEngine.run | query='%s'", query[:80])
 
        # ── Resolve contract name references ──────────────────────────────
        # If user says "abraxas agreement", find the matching indexed file
        resolved_file = source_file or self._resolve_contract_reference(
            query, known_files or []
        )
 
        enriched_query = self._enrich_with_memory(query, memory)
 
        # ── Retrieve ──────────────────────────────────────────────────────
        try:
            retrieval = self._retriever.retrieve(
                query=enriched_query,
                top_k=top_k,
                source_file=resolved_file,
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return QueryResponse(query=query, answer="Retrieval failed. Please try again.", is_grounded=False)
 
        # ── Generate ──────────────────────────────────────────────────────
        try:
            result: AnswerResult = self._generator.generate(
                query=query,
                context=retrieval.combined_text,
            )
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return QueryResponse(query=query, answer="Answer generation failed. Please try again.", is_grounded=False)
 
        # ── Memory ────────────────────────────────────────────────────────
        if memory is not None:
            try:
                memory.add(query=query, answer=result.answer, context=retrieval.combined_text)
            except Exception as exc:
                logger.warning("Memory update failed (non-critical): %s", exc)
 
        sources      = [chunk.text for chunk in retrieval.chunks]
        source_files = list({chunk.source_file for chunk in retrieval.chunks})
 
        return QueryResponse(
            query=query,
            answer=result.answer,
            sources=sources,
            source_files=source_files,
            is_out_of_scope=result.is_out_of_scope,
            is_grounded=result.is_grounded,
            retrieval=retrieval,
        )
 
    def summarise(self, context: str, memory=None) -> QueryResponse:
        try:
            result = self._generator.summarise(context)
        except Exception as exc:
            logger.error("Summarisation failed: %s", exc)
            return QueryResponse(query="Summarise the contract.", answer="Summarisation failed.", is_grounded=False)
 
        if memory is not None:
            try:
                memory.add(query="Summarise the contract.", answer=result.answer, context=context)
            except Exception as exc:
                logger.warning("Memory update failed: %s", exc)
 
        return QueryResponse(query="Summarise the contract.", answer=result.answer, is_grounded=True)
 
    # ── Private helpers ───────────────────────────────────────────────────
 
    def _resolve_contract_reference(
        self,
        query: str,
        known_files: list[str],
    ) -> str | None:
        """
        If the user's query references a contract by a short name or party name
        (e.g. "abraxas agreement", "concho contract"), find the best matching
        file from the list of known indexed files.
 
        Returns the matched filename, or None to search all files.
        """
        if not known_files:
            return None
 
        query_lower = query.lower()
 
        # Extract words from query — strip common contract words
        stop_words = {
            "the", "a", "an", "in", "of", "for", "and", "or", "is", "are",
            "what", "who", "when", "where", "which", "how", "agreement",
            "contract", "this", "that", "between", "parties", "party",
            "document", "deal", "terms",
        }
        query_words = {
            w for w in re.findall(r"[a-z0-9]+", query_lower)
            if w not in stop_words and len(w) > 2
        }
 
        if not query_words:
            return None
 
        best_file  = None
        best_score = 0
 
        for fname in known_files:
            # Build searchable tokens from filename — stem only, split on separators
            stem = Path(fname).stem.lower()
            file_tokens = set(re.findall(r"[a-z0-9]+", stem))
 
            overlap = query_words & file_tokens
            score   = len(overlap)
 
            logger.debug(
                "Contract match | file=%s | tokens=%s | query_words=%s | overlap=%s | score=%d",
                fname, file_tokens, query_words, overlap, score,
            )
 
            if score > best_score:
                best_score = score
                best_file  = fname
 
        # Only apply filter if at least 1 meaningful word matched
        if best_score >= 1:
            logger.info(
                "Resolved contract reference '%s' → '%s' (score=%d)",
                query[:60], best_file, best_score,
            )
            return best_file
 
        return None
 
    def _enrich_with_memory(self, query: str, memory) -> str:
        if memory is None:
            return query
        recent = memory.get_recent_context()
        if not recent:
            return query
        return f"{recent}\n\nCurrent question: {query}"
 
 
query_engine = QueryEngine()