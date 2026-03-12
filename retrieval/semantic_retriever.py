"""
retrieval/semantic_retriever.py
--------------------------------
LangChain 1.x compatible retriever wrapping Azure AI Search.
"""
 
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
 
from config.settings import settings
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store, SearchResult
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class RetrievedContext:
    query:         str
    chunks:        list
    combined_text: str
    top_k:         int
 
 
# ── LangChain BaseRetriever ────────────────────────────────────────────────
class ContractRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for Azure AI Search.
    Can be used directly in any LangChain chain.
    """
    top_k:       int = 5
    source_file: str = ""
 
    model_config = {"arbitrary_types_allowed": True}
 
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
 
        logger.info("ContractRetriever | query='%s' | filter='%s'", query[:60], self.source_file)
 
        query_vector = embedder.embed_query(query)
        results = vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k,
            source_file=self.source_file if self.source_file else None,
        )
 
        docs = [
            Document(
                page_content=r.text,
                metadata={
                    "source_file": r.source_file,
                    "score":       r.score,
                    "chunk_id":    r.chunk_id,
                },
            )
            for r in results
        ]
        logger.info("ContractRetriever returned %d docs", len(docs))
        return docs
 
 
# ── SemanticRetriever — preserves old interface ───────────────────────────
class SemanticRetriever:
    """
    Keeps the existing retrieve() interface so LangGraph nodes
    and the rest of the app need zero changes.
    Internally delegates to ContractRetriever (LangChain BaseRetriever).
    """
 
    def __init__(self) -> None:
        logger.info("SemanticRetriever initialised (LangChain 1.x backend).")
 
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> RetrievedContext:
 
        k          = top_k or settings.app.retrieval_top_k
        retriever  = ContractRetriever(top_k=k, source_file=source_file or "")
        docs       = retriever.invoke(query)
 
        chunks = [
            SearchResult(
                text=d.page_content,
                source_file=d.metadata.get("source_file", ""),
                score=d.metadata.get("score", 0.0),
                chunk_id=d.metadata.get("chunk_id", ""),
            )
            for d in docs
        ]
 
        parts = [
            f"[Excerpt {i} — Source: {r.source_file}]\n{r.text}"
            for i, r in enumerate(chunks, 1)
        ]
        combined_text = "\n\n---\n\n".join(parts)
 
        return RetrievedContext(
            query=query, chunks=chunks,
            combined_text=combined_text, top_k=k,
        )
 
    def as_langchain_retriever(
        self,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> ContractRetriever:
        """Return raw LangChain retriever for use in chains."""
        return ContractRetriever(
            top_k=top_k or settings.app.retrieval_top_k,
            source_file=source_file or "",
        )
 
 
semantic_retriever = SemanticRetriever()