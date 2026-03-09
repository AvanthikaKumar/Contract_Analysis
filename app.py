"""
app.py
------
Contract Intelligence System — single page layout.
Upload panel in sidebar, Q&A on main page.
"""
 
import logging
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
import streamlit as st
 
from core.memory import get_or_create_memory
from graph.entity_extractor import entity_extractor
from graph.graph_builder import graph_builder
from ingestion.chunker import chunker
from ingestion.document_loader import document_loader
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store
from retrieval.query_engine import query_engine
 
logger = logging.getLogger(__name__)
 
# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Intelligence System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
section[data-testid="stSidebar"] { min-width: 360px; max-width: 360px; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)
 
# ── Session state ──────────────────────────────────────────────────────────
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = {}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None
 
 
# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload & Processing (same step-by-step style as tab_upload.py)
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚖️ Contract Intelligence")
    st.markdown("AI-powered analysis using **GraphRAG** · Azure OpenAI · Cosmos DB")
    st.divider()
 
    st.header("📤 Upload Contracts")
    st.markdown("Upload one or more PDF contracts to extract text, generate embeddings, and build the knowledge graph.")
 
    uploaded_files = st.file_uploader(
        label="Upload Contract PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF contract files to process.",
    )
 
    new_files = [
        f for f in (uploaded_files or [])
        if f.name not in st.session_state["processed_files"]
    ]
 
    if uploaded_files and not new_files:
        st.info("✅ All uploaded files have already been processed.")
 
    if new_files:
        st.markdown(f"**{len(new_files)} new file(s) ready to process:**")
        for f in new_files:
            st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")
 
        if st.button("⚙️ Process Contracts", type="primary", use_container_width=True):
            memory = get_or_create_memory(st.session_state)
            memory.clear()
            st.session_state["chat_history"] = []
 
            for uf in new_files:
                st.markdown(f"---\n### Processing: `{uf.name}`")
                file_bytes = uf.read()
                result = {
                    "file_name": uf.name,
                    "page_count": 0, "char_count": 0,
                    "chunk_count": 0, "vectors_stored": 0,
                    "entities_found": 0, "relationships_found": 0,
                    "vertices_created": 0, "edges_created": 0,
                    "entity_list": [],
                    "error": None,
                }
 
                try:
                    # Step 1 — Extract text
                    with st.status("📄 Extracting text...", expanded=True) as status:
                        doc = document_loader.extract_text(file_bytes, uf.name)
                        result["page_count"] = doc.page_count
                        result["char_count"] = len(doc.full_text)
                        status.update(
                            label=f"✅ Text extracted — {doc.page_count} pages, {len(doc.full_text):,} characters",
                            state="complete",
                        )
 
                    # Step 2 — Chunk
                    with st.status("✂️ Chunking document...", expanded=True) as status:
                        chunks = chunker.split(doc)
                        result["chunk_count"] = len(chunks)
                        status.update(
                            label=f"✅ Document split into {len(chunks)} chunks",
                            state="complete",
                        )
 
                    # Step 3 — Embed
                    with st.status("🔢 Generating embeddings...", expanded=True) as status:
                        embedded_chunks = embedder.embed_chunks(chunks)
                        status.update(
                            label=f"✅ Embeddings generated for {len(embedded_chunks)} chunks",
                            state="complete",
                        )
 
                    # Step 4 — Store vectors
                    with st.status("💾 Storing in Azure AI Search...", expanded=True) as status:
                        count = vector_store.upload_chunks(embedded_chunks)
                        result["vectors_stored"] = count
                        status.update(
                            label=f"✅ {count} chunks stored in vector index",
                            state="complete",
                        )
 
                    # Step 5 — Extract entities
                    with st.status("🔍 Extracting entities...", expanded=True) as status:
                        extraction = entity_extractor.extract_from_chunks(
                            chunks, source_file=uf.name
                        )
                        result["entities_found"] = len(extraction.entities)
                        result["relationships_found"] = len(extraction.relationships)
                        result["entity_list"] = extraction.entities
                        status.update(
                            label=f"✅ Found {len(extraction.entities)} entities, {len(extraction.relationships)} relationships",
                            state="complete",
                        )
 
                    # Step 6 — Build graph
                    with st.status("🕸️ Building knowledge graph...", expanded=True) as status:
                        stats = graph_builder.build(extraction)
                        result["vertices_created"] = stats["vertices_created"]
                        result["edges_created"] = stats["edges_created"]
                        status.update(
                            label=f"✅ Graph updated — {stats['vertices_created']} vertices, {stats['edges_created']} edges",
                            state="complete",
                        )
 
                except Exception as exc:
                    logger.error("Pipeline failed for '%s': %s", uf.name, exc)
                    result["error"] = str(exc)
                    st.error(f"❌ Processing failed: {exc}")
 
                st.session_state["processed_files"][uf.name] = result
 
                if not result["error"]:
                    st.success(f"✅ **{uf.name}** processed successfully!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📄 Pages", result["page_count"])
                    c2.metric("✂️ Chunks", result["chunk_count"])
                    c3.metric("🔍 Entities", result["entities_found"])
                    c4, c5, c6 = st.columns(3)
                    c4.metric("💾 Vectors", result["vectors_stored"])
                    c5.metric("🕸️ Vertices", result["vertices_created"])
                    c6.metric("➡️ Edges", result["edges_created"])
 
            st.balloons()
            st.success("🎉 All contracts processed. Ask questions on the right →")
 
    # ── Processed files history ────────────────────────────────────────
    if st.session_state["processed_files"]:
        st.divider()
        st.subheader("📁 Processed in This Session")
        for fname, res in st.session_state["processed_files"].items():
            icon = "✅" if not res.get("error") else "❌"
            with st.expander(f"{icon} {fname}"):
                if res.get("error"):
                    st.error(res["error"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pages", res["page_count"])
                    c2.metric("Chunks", res["chunk_count"])
                    c3.metric("Entities", res["entities_found"])
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Vectors", res["vectors_stored"])
                    c5.metric("Vertices", res["vertices_created"])
                    c6.metric("Edges", res["edges_created"])
 
    # ── Example queries ────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Example Queries")
    st.caption("🔵 Direct  |  🟣 Relational (GraphRAG)")
 
    SUGGESTED = [
        ("🔵", "Who are the parties in this contract?"),
        ("🔵", "What is the contract effective date?"),
        ("🔵", "What are the payment terms?"),
        ("🔵", "What is the governing law?"),
        ("🟣", "How are the payment obligations connected to the seller's representations?"),
        ("🟣", "What clauses protect the buyer across both contracts?"),
        ("🟣", "Compare the termination conditions between the two contracts."),
    ]
    for icon, q in SUGGESTED:
        if st.button(f"{icon} {q}", key=f"sq_{q[:25]}", use_container_width=True):
            st.session_state["pending_query"] = q
            st.rerun()
 
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["chat_history"] = []
        get_or_create_memory(st.session_state).clear()
        st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════
# MAIN PAGE — Contract Intelligence Q&A
# ══════════════════════════════════════════════════════════════════════════
st.title("🔍 Contract Intelligence")
st.markdown(
    "Ask natural language questions about your uploaded contracts. "
    "The system retrieves relevant clauses and generates grounded answers."
)
st.divider()
 
# ── Guard ──────────────────────────────────────────────────────────────────
has_contracts = any(
    not r.get("error")
    for r in st.session_state["processed_files"].values()
)
if not has_contracts:
    st.warning("⚠️ No contracts indexed yet. Upload and process a contract in the left panel first.")
    st.stop()

# ── Metadata renderer ─────────────────────────────────────────────────────
def _render_metadata(meta: dict) -> None:
    """Show extracted metadata and supporting context below an answer."""
    entities    = meta.get("entities", [])
    sources     = meta.get("sources", [])
    src_files   = meta.get("source_files", [])
 
    if not entities and not sources and not src_files:
        return
 
    with st.expander("📊 Extracted Metadata & Supporting Context", expanded=False):
 
        # Entity metadata grouped by type
        if entities:
            st.markdown("**🔍 Relevant Graph Entities**")
            by_type: dict = {}
            for e in entities:
                by_type.setdefault(e["type"], []).append(e["label"])
 
            for etype, labels in by_type.items():
                st.markdown(f"**{etype.replace('_', ' ').title()}**")
                st.markdown(" · ".join(f"`{l}`" for l in labels))
 
        if src_files:
            st.markdown("---")
            st.markdown("**📁 Source Document(s)**")
            for sf in src_files:
                st.markdown(f"- `{sf}`")
 
        # Raw supporting excerpts
        if sources:
            st.markdown("---")
            st.markdown("**📄 Supporting Contract Excerpts**")
            for i, src in enumerate(sources[:3], 1):
                with st.expander(f"Excerpt {i}", expanded=False):
                    st.text(src[:600] + ("…" if len(src) > 600 else ""))
 
 
# ── Render existing chat ───────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_metadata(msg["meta"])
 
 
# ── Query processor ────────────────────────────────────────────────────────
def _process_query(query: str) -> None:
    memory = get_or_create_memory(st.session_state)
 
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["chat_history"].append({"role": "user", "content": query})
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass all successfully indexed filenames so the engine can
            # resolve "abraxas agreement" → actual filename automatically
            known_files = [
                fname for fname, res in st.session_state["processed_files"].items()
                if not res.get("error")
            ]
            response = query_engine.run(query=query, memory=memory, known_files=known_files)
 
        st.markdown(response.answer)
 
        # Only show entities whose label is explicitly mentioned in the
        # answer text or retrieved context — strict label matching only,
        # no type-based wildcards that bleed across documents
        answer_lower  = response.answer.lower()
        context_lower = " ".join(response.sources).lower()
        combined      = answer_lower + " " + context_lower
 
        relevant_entities = []
        for fname, res in st.session_state["processed_files"].items():
            if res.get("error"):
                continue
            for e in res.get("entity_list", []):
                if e.label.lower() in combined:
                    relevant_entities.append({"type": e.type, "label": e.label})
 
        # Deduplicate
        seen, deduped = set(), []
        for e in relevant_entities:
            k = (e["type"], e["label"])
            if k not in seen:
                seen.add(k)
                deduped.append(e)
 
        meta = {
            "entities":     deduped[:20],
            "sources":      response.sources,
            "source_files": response.source_files,
        }
        _render_metadata(meta)
 
        if response.is_out_of_scope:
            st.warning("⚠️ This question is outside contract scope.")
        elif not response.is_grounded:
            st.info("ℹ️ Answer could not be fully grounded in the contract context.")
 
        if response.source_files:
            st.caption("📁 Sources: " + ", ".join(f"`{f}`" for f in response.source_files))
 
    st.session_state["chat_history"].append({
        "role":    "assistant",
        "content": response.answer,
        "meta":    meta,
    })
 
 
# ── Pending query from sidebar buttons ────────────────────────────────────
if st.session_state["pending_query"]:
    q = st.session_state["pending_query"]
    st.session_state["pending_query"] = None
    _process_query(q)
 
# ── Chat input ─────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about your contract..."):
    _process_query(query)