"""
retrieval/langgraph_pipeline.py
--------------------------------
LangGraph 1.x compatible pipeline for Contract Intelligence System.
 
Graph nodes:
  1. scope_guard   — classify query as IN/OUT of scope
  2. retrieve      — vector search via Azure AI Search
  3. graph_lookup  — enrich context from Cosmos DB knowledge graph
  4. generate      — grounded answer via Azure OpenAI
  5. reject        — politely refuse out-of-scope queries
 
Flow:
  scope_guard → (out_of_scope) → reject → END
              → (in_scope)     → retrieve → graph_lookup → generate → END
"""
 
import logging
import re
import sys
from pathlib import Path
from typing import TypedDict
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
# LangGraph 1.x imports
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
 
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
 
 
# ── State schema ───────────────────────────────────────────────────────────
class ContractState(TypedDict):
    query:           str
    known_files:     list
    source_file:     str
    context:         str
    graph_context:   str
    answer:          str
    sources:         list
    source_files:    list
    is_out_of_scope: bool
    is_grounded:     bool
 
 
# ── Helper ─────────────────────────────────────────────────────────────────
def _resolve_contract_reference(query: str, known_files: list) -> str:
    if not known_files:
        return ""
    stop = {
        "the","a","an","in","of","for","and","or","is","are","what","who",
        "when","where","which","how","agreement","contract","this","that",
        "between","parties","party","document","deal","terms",
    }
    query_words = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in stop and len(w) > 2
    }
    best_file, best_score = "", 0
    for fname in known_files:
        tokens = set(re.findall(r"[a-z0-9]+", Path(fname).stem.lower()))
        score  = len(query_words & tokens)
        if score > best_score:
            best_score, best_file = score, fname
    return best_file if best_score >= 1 else ""
 
 
# ── Node 1: scope_guard ────────────────────────────────────────────────────
def scope_guard_node(state: ContractState) -> dict:
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
 
    source_file = _resolve_contract_reference(
        state["query"], state.get("known_files") or []
    )
 
    return {
        "is_out_of_scope": out_of_scope,
        "source_file":     source_file,
    }
 
 
# ── Node 2: reject ─────────────────────────────────────────────────────────
def reject_node(state: ContractState) -> dict:
    logger.info("[LangGraph] reject_node")
    return {
        "answer":       OUT_OF_SCOPE_RESPONSE,
        "is_grounded":  False,
        "sources":      [],
        "source_files": [],
        "context":      "",
        "graph_context":"",
    }
 
 
# ── Node 3: retrieve ──────────────────────────────────────────────────────
# Term synonym map — expands narrow queries to catch how contracts actually phrase things
_TERM_SYNONYMS = {
    "term end date":       "term end date expiration date end of term contract end termination date term until supply period ends through until 2039 2040 2041 2042 2043 2044 2045 2050",
    "end date":            "end date expiration term until through supply period end contract duration ends",
    "start date":          "start date commencement date effective date agreement date COD commercial operations date",
    "effective date":      "effective date commencement date start date execution date COD",
    "duration":            "duration term period supply period years until through commencing",
    "termination":         "termination cancellation end of agreement notice period",
    "payment terms":       "payment terms consideration purchase price compensation fees",
    "governing law":       "governing law jurisdiction applicable law choice of law",
    "parties":             "parties buyer seller purchaser vendor client contractor supplier",
    "purchase price":      "purchase price consideration total amount payment sum",
    "closing date":        "closing date completion date settlement date",
    "lng":                 "LNG liquefied natural gas supply delivered mtpa",
    "supply":              "supply deliver LNG mtpa volume quantity",
}
 
def _expand_query(query: str) -> str:
    """Expand query with synonyms so embedding captures contract language variations."""
    q_lower = query.lower()
    for term, expansion in _TERM_SYNONYMS.items():
        if term in q_lower:
            return query + " " + expansion
    return query
 
 
def retrieve_node(state: ContractState) -> dict:
    logger.info("[LangGraph] retrieve_node | file_filter='%s'", state.get("source_file", ""))
    try:
        source_file    = state.get("source_file") or None
        known_files    = state.get("known_files") or []
        expanded_query = _expand_query(state["query"])
        query_vector   = embedder.embed_query(expanded_query)
 
        # Detect compare / cross-document intent
        compare_keywords = [
            "compare", "both", "difference", "versus", "vs",
            "across", "between", "all contracts", "both contracts",
        ]
        is_compare = any(kw in state["query"].lower() for kw in compare_keywords)
 
        if is_compare and len(known_files) >= 2 and not source_file:
            # Fetch chunks from EACH contract separately so no contract is skipped
            all_results = []
            per_file_k  = max(3, settings.app.retrieval_top_k // len(known_files))
            for fname in known_files:
                file_results = vector_store.search(
                    query_vector=query_vector,
                    top_k=per_file_k,
                    source_file=fname,
                )
                all_results.extend(file_results)
                logger.info(
                    "[LangGraph] retrieve_node | file=%s | chunks=%d", fname, len(file_results)
                )
            results = all_results
        else:
            results = vector_store.search(
                query_vector=query_vector,
                top_k=settings.app.retrieval_top_k,
                source_file=source_file if source_file else None,
            )
 
        parts        = [f"[Excerpt {i} — Source: {r.source_file}]\n{r.text}" for i, r in enumerate(results, 1)]
        context      = "\n\n---\n\n".join(parts)
        sources      = [r.text for r in results]
        source_files = list({r.source_file for r in results})
        logger.info("[LangGraph] retrieve_node | total_chunks=%d", len(results))
 
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        context, sources, source_files = "", [], []
 
    return {"context": context, "sources": sources, "source_files": source_files}

# ── Node 4: graph_lookup ──────────────────────────────────────────────────
# Question → intent → graph traversal path → structured answer
_INTENT_TO_TRAVERSAL = {
    # Direct lookups
    "parties":        ("PARTY",        ["PARTY_TO_CONTRACT","PARTY_IS_BUYER","PARTY_IS_SELLER"]),
    "buyer":          ("PARTY",        ["PARTY_IS_BUYER"]),
    "seller":         ("PARTY",        ["PARTY_IS_SELLER"]),
    "supplier":       ("PARTY",        ["PARTY_IS_SELLER","PARTY_SUPPLIES"]),
    "date":           ("DATE",         ["EFFECTIVE_FROM","EXPIRES_ON"]),
    "start date":     ("DATE",         ["EFFECTIVE_FROM"]),
    "end date":       ("DATE",         ["EXPIRES_ON"]),
    "term":           ("DATE",         ["EXPIRES_ON","EFFECTIVE_FROM"]),
    "expiry":         ("DATE",         ["EXPIRES_ON"]),
    "governing law":  ("GOVERNING_LAW",["GOVERNED_BY","PARTY_GOVERNED_BY"]),
    "jurisdiction":   ("GOVERNING_LAW",["GOVERNED_BY"]),
    "obligation":     ("OBLIGATION",   ["PARTY_HAS_OBLIGATION","CLAUSE_CONTAINS_OBLIGATION"]),
    "payment":        ("FINANCIAL_TERM",["CLAUSE_CONTAINS_FINANCIAL_TERM","FINANCIAL_TERM_IN_CLAUSE"]),
    "penalty":        ("FINANCIAL_TERM",["OBLIGATION_TRIGGERS_FINANCIAL_TERM"]),
    "supply":         ("PRODUCT",      ["PARTY_SUPPLIES","PARTY_RECEIVES"]),
    "deliver":        ("PRODUCT",      ["PARTY_SUPPLIES","DELIVERY_LOCATION"]),
    "lng":            ("PRODUCT",      ["PARTY_SUPPLIES","PARTY_RECEIVES","DELIVERY_LOCATION"]),
    "volume":         ("PRODUCT",      ["PARTY_SUPPLIES"]),
    "quantity":       ("PRODUCT",      ["PARTY_SUPPLIES"]),
    "clause":         ("CLAUSE",       ["CLAUSE_CONTAINS_OBLIGATION","CLAUSE_CONTAINS_FINANCIAL_TERM"]),
    "location":       ("LOCATION",     ["DELIVERY_LOCATION"]),
}
 
def _detect_intent(query: str) -> list[tuple]:
    """Match query keywords to graph traversal intents."""
    q = query.lower()
    matched = []
    for keyword, (entity_type, edge_types) in _INTENT_TO_TRAVERSAL.items():
        if keyword in q:
            matched.append((keyword, entity_type, edge_types))
    return matched
 
 
def _traverse_graph(contract_id: str, entity_type: str, edge_types: list[str]) -> list[dict]:
    """
    Traverse the graph:
      CONTRACT hub → edge_type → entity of entity_type
    Returns list of {label, properties, edge_type, context}
    """
    results = []
    for edge_type in edge_types:
        try:
            # Outgoing: CONTRACT → edge → entity
            rows = graph_client.execute(
                "g.V().hasId(cid).outE(etype).as('e')"
                ".inV().has('entity_type', etype2).as('v')"
                ".select('e','v')"
                ".by(valueMap(true))"
                ".by(valueMap(true))",
                bindings={
                    "cid":   contract_id,
                    "etype": edge_type,
                    "etype2": entity_type,
                },
            )
            for row in (rows or []):
                v = row.get("v", {})
                e = row.get("e", {})
                label = v.get("entity_label", [""])[0] if isinstance(v.get("entity_label"), list) else v.get("entity_label", "")
                props = {}
                for k, val in v.items():
                    if k not in ("id", "pk", "entity_label", "entity_type", "source_file", "entity_id"):
                        props[k] = val[0] if isinstance(val, list) else val
                edge_props = {}
                for k, val in e.items():
                    if k not in ("id",):
                        edge_props[k] = val[0] if isinstance(val, list) else val
 
                if label:
                    results.append({
                        "label":      label,
                        "properties": props,
                        "edge_type":  edge_type,
                        "context":    edge_props.get("context", ""),
                    })
        except Exception as exc:
            logger.debug("Traversal %s→%s→%s failed: %s", contract_id, edge_type, entity_type, exc)
 
        # Also try reverse: entity → edge → CONTRACT
        try:
            rows = graph_client.execute(
                "g.V().hasId(cid).inE(etype).as('e')"
                ".outV().has('entity_type', etype2).as('v')"
                ".select('e','v')"
                ".by(valueMap(true))"
                ".by(valueMap(true))",
                bindings={
                    "cid":    contract_id,
                    "etype":  edge_type,
                    "etype2": entity_type,
                },
            )
            for row in (rows or []):
                v = row.get("v", {})
                e = row.get("e", {})
                label = v.get("entity_label", [""])[0] if isinstance(v.get("entity_label"), list) else v.get("entity_label", "")
                props = {}
                for k, val in v.items():
                    if k not in ("id", "pk", "entity_label", "entity_type", "source_file", "entity_id"):
                        props[k] = val[0] if isinstance(val, list) else val
                if label:
                    results.append({
                        "label":     label,
                        "properties": props,
                        "edge_type": edge_type,
                        "context":   "",
                    })
        except Exception:
            pass
 
    return results
 
 
def graph_lookup_node(state: ContractState) -> dict:
    """
    Real GraphRAG — answer the question by traversing relationships.
 
    Instead of dumping all entities, we:
    1. Detect what the question is asking for (intent)
    2. Map intent to specific edge types to traverse
    3. Traverse CONTRACT → edge → entity
    4. Build a structured answer from the graph paths found
    """
    logger.info("[LangGraph] graph_lookup_node (relationship traversal)")
    graph_parts = []
 
    try:
        source_file  = state.get("source_file") or ""
        source_files = state.get("source_files") or []
        files_to_query = [source_file] if source_file else source_files[:2]
 
        # Detect intents from the question
        intents = _detect_intent(state["query"])
        logger.info("[LangGraph] graph intents detected: %s", [i[0] for i in intents])
 
        if not intents:
            # No specific intent — fall back to all entities for this contract
            intents = [("all", "PARTY", ["PARTY_TO_CONTRACT"]),
                       ("all", "DATE",  ["EFFECTIVE_FROM","EXPIRES_ON"])]
 
        for fname in files_to_query:
            if not fname:
                continue
            contract_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(fname).stem).strip("_")[:200]
            file_parts  = [f"\n[Graph traversal for: {fname}]"]
 
            for keyword, entity_type, edge_types in intents:
                results = _traverse_graph(contract_id, entity_type, edge_types)
                if results:
                    file_parts.append(f"  {entity_type} ({keyword}):")
                    for r in results:
                        line = f"    → {r['label']}"
                        # Add properties — these carry the real answer values
                        if r["properties"]:
                            props_str = ", ".join(
                                f"{k}: {v}" for k, v in r["properties"].items()
                                if v and k not in ("prop_key_0","prop_val_0")
                            )
                            if props_str:
                                line += f" [{props_str}]"
                        if r["context"]:
                            line += f' (contract says: "{r["context"]}")'
                        file_parts.append(line)
 
            if len(file_parts) > 1:
                graph_parts.extend(file_parts)
 
        # For cross-document / compare questions — traverse both contracts
        # and show how they differ
        compare_keywords = ["compare","both","difference","versus","vs","across","between"]
        if any(kw in state["query"].lower() for kw in compare_keywords) and len(files_to_query) >= 2:
            graph_parts.append("\n[Cross-document relationship comparison]")
            for intent_keyword, entity_type, edge_types in intents:
                graph_parts.append(f"  {entity_type}:")
                for fname in files_to_query:
                    contract_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(fname).stem).strip("_")[:200]
                    results = _traverse_graph(contract_id, entity_type, edge_types)
                    labels = [r["label"] for r in results]
                    if labels:
                        graph_parts.append(f"    {fname}: {', '.join(labels)}")
 
    except Exception as exc:
        logger.error("graph_lookup_node error: %s", exc)
 
    graph_context = "\n".join(graph_parts)
    logger.info("[LangGraph] graph_lookup_node | chars=%d", len(graph_context))
    return {"graph_context": graph_context}
 
 
# ── Node 5: generate ──────────────────────────────────────────────────────
def generate_node(state: ContractState) -> dict:
    logger.info("[LangGraph] generate_node")
 
    full_context = state.get("context", "")
    if state.get("graph_context"):
        full_context += "\n\n=== Knowledge Graph Context ===\n" + state["graph_context"]
 
    if not full_context.strip():
        return {"answer": OUT_OF_CONTEXT_RESPONSE, "is_grounded": False}
 
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
                        "Answer strictly from the provided context. "
                        f"If not found, respond: '{OUT_OF_CONTEXT_RESPONSE}'"
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
 
    return {"answer": answer, "is_grounded": is_grounded}
 
 
# ── Routing ────────────────────────────────────────────────────────────────
def route_after_scope(state: ContractState) -> str:
    return "reject" if state.get("is_out_of_scope") else "retrieve"

# ── Build graph ────────────────────────────────────────────────────────────
def build_contract_graph():
    builder = StateGraph(ContractState)
 
    builder.add_node("scope_guard",  scope_guard_node)
    builder.add_node("reject",       reject_node)
    builder.add_node("retrieve",     retrieve_node)
    builder.add_node("graph_lookup", graph_lookup_node)
    builder.add_node("generate",     generate_node)
 
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
 
    # LangGraph 1.x — MemorySaver passed to compile()
    return builder.compile(checkpointer=MemorySaver())
 
 
contract_graph = build_contract_graph()
logger.info("LangGraph 1.x contract pipeline compiled successfully.")