"""
retrieval/langgraph_pipeline.py
--------------------------------
LangGraph-based query pipeline for the Contract Intelligence System.
 
This replaces the custom QueryEngine with a proper LangGraph StateGraph.
 
Graph nodes:
  1. scope_guard   — classify query as IN/OUT of scope
  2. retrieve      — vector search via Azure AI Search
  3. graph_lookup  — enrich context from Cosmos DB knowledge graph
  4. generate      — grounded answer via Azure OpenAI
  5. memory_save   — persist turn to LangGraph MemorySaver
  6. reject        — politely refuse out-of-scope queries
 
Flow:
  scope_guard → (out_of_scope) → reject
              → (in_scope)     → retrieve → graph_lookup → generate → memory_save
"""
 
import logging
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from langgraph.graph import END, START, StateGraph
from core.memory import langgraph_memory
 
from config.settings import settings
from core.prompt_manager import prompt_manager
from graph.graph_client import graph_client
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store
from llm.azure_openai_client import openai_client
 
logger = logging.getLogger(__name__)
 
OUT_OF_CONTEXT_RESPONSE = "Not specified in the provided document."
OUT_OF_SCOPE_RESPONSE = (
    "This question is outside the scope of contract analysis. "
    "Please ask a question related to the uploaded contract."
)
 
 
# ── State ─────────────────────────────────────────────────────────────────
class ContractState(TypedDict):
    """Shared state passed between all LangGraph nodes."""
    query:           str
    known_files:     list[str]
    source_file:     str | None       # resolved contract file filter
    context:         str              # vector search results
    graph_context:   str              # Cosmos DB graph enrichment
    answer:          str
    sources:         list[str]
    source_files:    list[str]
    is_out_of_scope: bool
    is_grounded:     bool
 
 
# ── Helper — resolve contract name reference ───────────────────────────────
def _resolve_contract_reference(query: str, known_files: list[str]) -> str | None:
    if not known_files:
        return None
    stop = {
        "the","a","an","in","of","for","and","or","is","are","what","who",
        "when","where","which","how","agreement","contract","this","that",
        "between","parties","party","document","deal","terms",
    }
    query_words = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in stop and len(w) > 2
    }
    best_file, best_score = None, 0
    for fname in known_files:
        tokens = set(re.findall(r"[a-z0-9]+", Path(fname).stem.lower()))
        score  = len(query_words & tokens)
        if score > best_score:
            best_score, best_file = score, fname
    return best_file if best_score >= 1 else None
 
 
# ── Node 1: scope_guard ────────────────────────────────────────────────────
def scope_guard_node(state: ContractState) -> ContractState:
    """Classify query as IN_SCOPE or OUT_OF_SCOPE."""
    logger.info("[LangGraph] scope_guard | query='%s'", state["query"][:60])
    try:
        prompt = prompt_manager.load(
            "scope_guard_prompt",
            variables={"question": state["query"]},
        )
        response = openai_client.get_chat_completion(
            messages=[
                {"role": "system", "content": "You are a query classifier. Respond with only IN_SCOPE or OUT_OF_SCOPE."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        out_of_scope = "OUT_OF_SCOPE" in response.strip().upper()
    except Exception as exc:
        logger.warning("Scope check failed — defaulting IN_SCOPE: %s", exc)
        out_of_scope = False
 
    # Resolve which contract file the query refers to
    source_file = _resolve_contract_reference(state["query"], state.get("known_files", []))
 
    return {
        **state,
        "is_out_of_scope": out_of_scope,
        "source_file":     source_file,
    }
 
 
# ── Node 2: reject ─────────────────────────────────────────────────────────
def reject_node(state: ContractState) -> ContractState:
    """Return polite refusal for out-of-scope queries."""
    logger.info("[LangGraph] reject_node triggered")
    return {
        **state,
        "answer":      OUT_OF_SCOPE_RESPONSE,
        "is_grounded": False,
        "sources":     [],
        "source_files":[],
    }
 
 
# ── Node 3: retrieve ──────────────────────────────────────────────────────
def retrieve_node(state: ContractState) -> ContractState:
    """Vector search Azure AI Search for relevant contract chunks."""
    logger.info("[LangGraph] retrieve_node | file_filter=%s", state.get("source_file"))
    try:
        query_vector = embedder.embed_query(state["query"])
        results = vector_store.search(
            query_vector=query_vector,
            top_k=settings.app.retrieval_top_k,
            source_file=state.get("source_file"),
        )
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Excerpt {i} — Source: {r.source_file}]\n{r.text}")
 
        context      = "\n\n---\n\n".join(parts)
        sources      = [r.text for r in results]
        source_files = list({r.source_file for r in results})
 
        logger.info("[LangGraph] retrieve_node | chunks=%d", len(results))
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        context, sources, source_files = "", [], []
 
    return {**state, "context": context, "sources": sources, "source_files": source_files}
 
 
# ── Node 4: graph_lookup ──────────────────────────────────────────────────
def graph_lookup_node(state: ContractState) -> ContractState:
    """
    Enrich context with Cosmos DB knowledge graph data.
 
    THIS is what makes it real GraphRAG — we query the graph at
    question-answering time, not just at ingestion time.
 
    Pulls:
    - All parties connected to the relevant contract hub
    - Dates, governing law, financial terms
    - Entity relationships extracted by the LLM during ingestion
    """
    logger.info("[LangGraph] graph_lookup_node")
    graph_parts = []
 
    try:
        source_files = state.get("source_files", [])
        files_to_query = (
            [state["source_file"]] if state.get("source_file")
            else source_files[:2]   # limit to top 2 contracts
        )
 
        for fname in files_to_query:
            from pathlib import Path as _P
            import re as _re
            contract_id = _re.sub(r"[^a-zA-Z0-9_\-]", "_", _P(fname).stem).strip("_")[:200]
 
            # Get all entities connected to this contract hub
            try:
                rows = graph_client.execute(
                    "g.V().hasId(cid).out().valueMap(true)",
                    bindings={"cid": contract_id},
                )
                if rows:
                    graph_parts.append(f"\n[Graph entities for: {fname}]")
                    by_type: dict = {}
                    for row in rows:
                        etype  = row.get("entity_type", ["UNKNOWN"])[0] if isinstance(row.get("entity_type"), list) else row.get("entity_type", "UNKNOWN")
                        elabel = row.get("entity_label", [""])[0]       if isinstance(row.get("entity_label"), list) else row.get("entity_label", "")
                        if elabel:
                            by_type.setdefault(etype, []).append(elabel)
 
                    for etype, labels in by_type.items():
                        graph_parts.append(f"  {etype}: {', '.join(labels)}")
 
            except Exception as exc:
                logger.warning("Graph lookup failed for '%s': %s", contract_id, exc)
 
        # If query seems relational, also get edges
        relational_keywords = ["connect", "related", "between", "compare", "both", "across", "share"]
        if any(kw in state["query"].lower() for kw in relational_keywords):
            try:
                edge_rows = graph_client.execute(
                    "g.E().limit(50).project('from','label','to')"
                    ".by(__.outV().values('entity_label'))"
                    ".by(__.label())"
                    ".by(__.inV().values('entity_label'))",
                    bindings={},
                )
                if edge_rows:
                    graph_parts.append("\n[Graph relationships]")
                    for edge in edge_rows[:20]:
                        frm = edge.get("from", "")
                        lbl = edge.get("label", "")
                        to  = edge.get("to",   "")
                        if frm and lbl and to:
                            graph_parts.append(f"  {frm} --[{lbl}]--> {to}")
            except Exception as exc:
                logger.warning("Edge lookup failed: %s", exc)
 
    except Exception as exc:
        logger.error("graph_lookup_node failed: %s", exc)
 
    graph_context = "\n".join(graph_parts) if graph_parts else ""
    logger.info("[LangGraph] graph_lookup_node | graph_chars=%d", len(graph_context))
 
    return {**state, "graph_context": graph_context}
 
 
# ── Node 5: generate ──────────────────────────────────────────────────────
def generate_node(state: ContractState) -> ContractState:
    """Generate grounded answer combining vector context + graph context."""
    logger.info("[LangGraph] generate_node")
 
    # Combine vector context with graph enrichment
    full_context = state.get("context", "")
    if state.get("graph_context"):
        full_context = (
            full_context
            + "\n\n=== Knowledge Graph Context ===\n"
            + state["graph_context"]
        )
 
    if not full_context.strip():
        return {
            **state,
            "answer":      OUT_OF_CONTEXT_RESPONSE,
            "is_grounded": False,
        }
 
    try:
        prompt = prompt_manager.load(
            "answer_prompt",
            variables={"context": full_context, "question": state["query"]},
        )
        answer = openai_client.get_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise contract analysis assistant. "
                        "Answer questions strictly based on the provided context. "
                        f"If the answer is not in the context, respond with: '{OUT_OF_CONTEXT_RESPONSE}'"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        answer      = answer.strip()
        is_grounded = OUT_OF_CONTEXT_RESPONSE.lower() not in answer.lower()
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer, is_grounded = "Answer generation failed. Please try again.", False
 
    return {**state, "answer": answer, "is_grounded": is_grounded}

# ── Routing function ──────────────────────────────────────────────────────
def route_after_scope(state: ContractState) -> str:
    return "reject" if state["is_out_of_scope"] else "retrieve"
 
 
# ── Build the graph ───────────────────────────────────────────────────────
def build_contract_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(ContractState)
 
    # Add nodes
    builder.add_node("scope_guard",  scope_guard_node)
    builder.add_node("reject",       reject_node)
    builder.add_node("retrieve",     retrieve_node)
    builder.add_node("graph_lookup", graph_lookup_node)
    builder.add_node("generate",     generate_node)
 
    # Add edges
    builder.add_edge(START, "scope_guard")
    builder.add_conditional_edges(
        "scope_guard",
        route_after_scope,
        {"reject": "reject", "retrieve": "retrieve"},
    )
    builder.add_edge("reject",       END)
    builder.add_edge("retrieve",     "graph_lookup")
    builder.add_edge("graph_lookup", "generate")
    builder.add_edge("generate",     END)
 
    return builder.compile(checkpointer=langgraph_memory)
 
 
# ── Compiled graph singleton ──────────────────────────────────────────────
contract_graph = build_contract_graph()
logger.info("LangGraph contract pipeline compiled successfully.")