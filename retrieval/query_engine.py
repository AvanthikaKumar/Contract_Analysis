"""
retrieval/query_engine.py
--------------------------
QueryEngine — now powered by LangGraph.
 
The actual pipeline lives in langgraph_pipeline.py.
This file is a thin wrapper that keeps the same interface
the UI (app.py) already uses, so no UI changes are needed.
"""
 
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from retrieval.langgraph_pipeline import contract_graph, ContractState
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store
from llm.answer_generator import answer_generator
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class QueryResponse:
    query:           str
    answer:          str
    sources:         list[str] = field(default_factory=list)
    source_files:    list[str] = field(default_factory=list)
    is_out_of_scope: bool = False
    is_grounded:     bool = True
    retrieval:       object = None
 
 
class QueryEngine:
 
    def __init__(self) -> None:
        # Each session gets its own thread_id for LangGraph MemorySaver
        self._thread_id = "default-session"
        logger.info("QueryEngine initialised (LangGraph backend).")
 
    def set_thread_id(self, thread_id: str) -> None:
        """Call this with a unique ID per user session."""
        self._thread_id = thread_id
 
    def run(
        self,
        query: str,
        memory=None,
        source_file: str | None = None,
        top_k: int | None = None,
        known_files: list[str] | None = None,
        use_graph: bool = True,
    ) -> QueryResponse:
 
        if not query or not query.strip():
            return QueryResponse(query=query, answer="Please enter a question.", is_grounded=False)
 
        logger.info(
            "QueryEngine.run | query='%s' | mode=%s",
            query[:80], "GraphRAG" if use_graph else "RAG",
        )
 
        # Build initial state
        initial_state: ContractState = {
            "query":           query,
            "known_files":     known_files or [],
            "source_file":     source_file,
            "context":         "",
            "graph_context":   "",
            "answer":          "",
            "sources":         [],
            "source_files":    [],
            "is_out_of_scope": False,
            "is_grounded":     True,
            "use_graph":       use_graph,
        }
 
        # Run through LangGraph — thread_id enables per-session memory
        config = {"configurable": {"thread_id": self._thread_id}}
 
        try:
            final_state = contract_graph.invoke(initial_state, config=config)
        except Exception as exc:
            logger.error("LangGraph pipeline failed: %s", exc)
            return QueryResponse(
                query=query,
                answer="Pipeline error. Please try again.",
                is_grounded=False,
            )
 
        # Also update legacy memory if passed (keeps UI memory display working)
        if memory is not None:
            try:
                memory.add(
                    query=query,
                    answer=final_state["answer"],
                    context=final_state.get("context", ""),
                )
            except Exception as exc:
                logger.warning("Legacy memory update failed: %s", exc)
 
        return QueryResponse(
            query=query,
            answer=final_state["answer"],
            sources=final_state.get("sources", []),
            source_files=final_state.get("source_files", []),
            is_out_of_scope=final_state.get("is_out_of_scope", False),
            is_grounded=final_state.get("is_grounded", True),
        )
 
    def summarise(self, context: str, memory=None) -> QueryResponse:
        """Summarise — uses existing answer_generator directly (no graph needed)."""
        try:
            result = answer_generator.summarise(context)
        except Exception as exc:
            logger.error("Summarisation failed: %s", exc)
            return QueryResponse(query="Summarise the contract.", answer="Summarisation failed.", is_grounded=False)
 
        if memory is not None:
            try:
                memory.add(query="Summarise the contract.", answer=result.answer, context=context)
            except Exception:
                pass
 
        return QueryResponse(
            query="Summarise the contract.",
            answer=result.answer,
            is_grounded=True,
        )
 
 
query_engine = QueryEngine()
 