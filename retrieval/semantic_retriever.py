"""
retrieval/semantic_retriever.py
--------------------------------
LangChain-wrapped retriever for Azure AI Search.
 
Replaces the custom vector search with a LangChain BaseRetriever
so it plugs directly into LangChain/LangGraph chains.
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
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
 
from config.settings import settings
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store, SearchResult
 
logger = logging.getLogger(__name__)
 
 
# ── Keep original dataclass so LangGraph pipeline stays compatible ─────────
@dataclass
class RetrievedContext:
    query:         str
    chunks:        list[SearchResult]
    combined_text: str
    top_k:         int
 
 
# ── LangChain BaseRetriever implementation ─────────────────────────────────
class ContractRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that searches Azure AI Search
    using vector embeddings.
 
    Wraps our existing vector_store so it can be used in any
    LangChain chain or LangGraph node directly.
    """
 
    top_k:       int = 5
    source_file: str | None = None
 
    class Config:
        arbitrary_types_allowed = True
 
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Required LangChain method.
        Embeds the query and returns matching contract chunks as Documents.
        """
        logger.info(
            "ContractRetriever._get_relevant_documents | query='%s' | filter=%s",
            query[:60], self.source_file,
        )
 
        query_vector = embedder.embed_query(query)
        results = vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k,
            source_file=self.source_file,
        )
 
        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.text,
                metadata={
                    "source_file": r.source_file,
                    "score":       r.score,
                    "chunk_id":    r.chunk_id,
                },
            ))
 
        logger.info("ContractRetriever returned %d documents", len(docs))
        return docs
 
 
# ── SemanticRetriever — keeps old interface for LangGraph nodes ────────────
class SemanticRetriever:
    """
    Thin wrapper that keeps the existing retrieve() interface
    while using ContractRetriever internally.
 
    This means the LangGraph pipeline (retrieve_node) needs
    zero changes — it still calls semantic_retriever.retrieve().
    """
 
    def __init__(self) -> None:
        logger.info("SemanticRetriever initialised (LangChain backend).")
 
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> RetrievedContext:
 
        k = top_k or settings.app.retrieval_top_k
 
        # Build a ContractRetriever with the right params and invoke it
        retriever = ContractRetriever(top_k=k, source_file=source_file)
        docs = retriever.invoke(query)
 
        # Convert LangChain Documents back to SearchResult for compatibility
        chunks = []
        for doc in docs:
            chunks.append(SearchResult(
                text=doc.page_content,
                source_file=doc.metadata.get("source_file", ""),
                score=doc.metadata.get("score", 0.0),
                chunk_id=doc.metadata.get("chunk_id", ""),
            ))
 
        combined_text = self._build_combined_text(chunks)
 
        return RetrievedContext(
            query=query,
            chunks=chunks,
            combined_text=combined_text,
            top_k=k,
        )
 
    def _build_combined_text(self, results: list[SearchResult]) -> str:
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Excerpt {i} — Source: {r.source_file}]\n{r.text}")
        return "\n\n---\n\n".join(parts)
 
    def as_langchain_retriever(
        self,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> ContractRetriever:
        """
        Return a raw LangChain BaseRetriever for direct use in chains.
        e.g. retrieval_chain = retriever | prompt | llm
        """
        return ContractRetriever(
            top_k=top_k or settings.app.retrieval_top_k,
            source_file=source_file,
        )
 
 
semantic_retriever = SemanticRetriever()