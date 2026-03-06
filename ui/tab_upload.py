"""
ui/tab_upload.py
-----------------
Streamlit Tab 1 — Contract Upload & Processing.
 
Responsibilities:
- Accept PDF uploads from the user.
- Run the full ingestion pipeline on uploaded files.
- Display extraction metadata, chunk counts, and graph node counts.
- Store ingestion results in session state for Tab 2 to reference.
 
This module contains only UI logic. All business logic is
delegated to the ingestion and graph modules.
"""
 
import logging
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
_KEY_PROCESSED_FILES = "processed_files"
_KEY_LAST_INGESTION  = "last_ingestion_result"
 
 
# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def _run_ingestion_pipeline(
    file_bytes: bytes,
    file_name: str,
) -> dict:
    """
    Run the full ingestion pipeline for one uploaded PDF.
 
    Returns a dict of results for display in the UI.
    """
    results = {
        "file_name":        file_name,
        "page_count":       0,
        "char_count":       0,
        "chunk_count":      0,
        "vectors_stored":   0,
        "entities_found":   0,
        "relationships_found": 0,
        "vertices_created": 0,
        "edges_created":    0,
        "error":            None,
    }
 
    try:
        # Step 1 — Extract text
        with st.status("📄 Extracting text...", expanded=True) as status:
            doc = document_loader.extract_text(file_bytes, file_name)
            results["page_count"] = doc.page_count
            results["char_count"] = len(doc.full_text)
            status.update(
                label=f"✅ Text extracted — {doc.page_count} pages, "
                      f"{len(doc.full_text):,} characters",
                state="complete",
            )
 
        # Step 2 — Chunk
        with st.status("✂️ Chunking document...", expanded=True) as status:
            chunks = chunker.split(doc)
            results["chunk_count"] = len(chunks)
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
            results["vectors_stored"] = count
            status.update(
                label=f"✅ {count} chunks stored in vector index",
                state="complete",
            )
 
        # Step 5 — Extract entities
        with st.status("🔍 Extracting entities...", expanded=True) as status:
            extraction = entity_extractor.extract_from_chunks(
                chunks, source_file=file_name
            )
            results["entities_found"]      = len(extraction.entities)
            results["relationships_found"] = len(extraction.relationships)
            status.update(
                label=f"✅ Found {len(extraction.entities)} entities, "
                      f"{len(extraction.relationships)} relationships",
                state="complete",
            )
 
        # Step 6 — Build graph
        with st.status("🕸️ Building knowledge graph...", expanded=True) as status:
            stats = graph_builder.build(extraction)
            results["vertices_created"] = stats["vertices_created"]
            results["edges_created"]    = stats["edges_created"]
            status.update(
                label=f"✅ Graph updated — {stats['vertices_created']} vertices, "
                      f"{stats['edges_created']} edges created",
                state="complete",
            )
 
    except Exception as exc:
        logger.error("Ingestion pipeline failed for '%s': %s", file_name, exc)
        results["error"] = str(exc)
 
    return results
 
 
# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
def _display_results(result: dict) -> None:
    """Render ingestion result metrics in the UI."""
 
    if result["error"]:
        st.error(f"❌ Processing failed: {result['error']}")
        return
 
    st.success(f"✅ **{result['file_name']}** processed successfully!")
 
    # Metrics row 1 — Document stats
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Pages",      result["page_count"])
    c2.metric("🔤 Characters", f"{result['char_count']:,}")
    c3.metric("✂️ Chunks",     result["chunk_count"])
 
    # Metrics row 2 — Storage stats
    c4, c5, c6 = st.columns(3)
    c4.metric("💾 Vectors Stored",  result["vectors_stored"])
    c5.metric("🔍 Entities Found",  result["entities_found"])
    c6.metric("🔗 Relationships",   result["relationships_found"])
 
    # Metrics row 3 — Graph stats
    c7, c8, _ = st.columns(3)
    c7.metric("🕸️ Vertices Created", result["vertices_created"])
    c8.metric("➡️ Edges Created",    result["edges_created"])
 
 
# ---------------------------------------------------------------------------
# Main tab render function
# ---------------------------------------------------------------------------
def render_upload_tab() -> None:
    """Render the full Upload & Processing tab."""
 
    st.header("📤 Contract Upload & Processing")
    st.markdown(
        "Upload one or more PDF contracts to extract text, generate embeddings, "
        "and build the knowledge graph."
    )
 
    # Initialise session state
    if _KEY_PROCESSED_FILES not in st.session_state:
        st.session_state[_KEY_PROCESSED_FILES] = {}
 
    # -------------------------------------------------------------------
    # File uploader
    # -------------------------------------------------------------------
    uploaded_files = st.file_uploader(
        label="Upload Contract PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF contract files to process.",
    )
 
    if not uploaded_files:
        st.info("👆 Upload a PDF contract above to begin.")
        _render_processed_history()
        return
 
    # -------------------------------------------------------------------
    # Process button
    # -------------------------------------------------------------------
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state[_KEY_PROCESSED_FILES]
    ]
 
    if not new_files:
        st.info("✅ All uploaded files have already been processed.")
        _render_processed_history()
        return
 
    st.markdown(f"**{len(new_files)} new file(s) ready to process:**")
    for f in new_files:
        st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")
 
    if st.button(
        "⚙️ Process Contracts",
        type="primary",
        use_container_width=True,
    ):
        # Clear memory when new contracts are processed
        memory = get_or_create_memory(st.session_state)
        memory.clear()
 
        for uploaded_file in new_files:
            st.markdown(f"---\n### Processing: `{uploaded_file.name}`")
 
            file_bytes = uploaded_file.read()
            result = _run_ingestion_pipeline(file_bytes, uploaded_file.name)
 
            # Store result in session state
            st.session_state[_KEY_PROCESSED_FILES][uploaded_file.name] = result
            st.session_state[_KEY_LAST_INGESTION] = result
 
            # Display results
            _display_results(result)
 
        st.balloons()
        st.success(
            "🎉 All contracts processed. Switch to the "
            "**Contract Intelligence** tab to ask questions."
        )
 
    # -------------------------------------------------------------------
    # Processed files history
    # -------------------------------------------------------------------
    _render_processed_history()
 
 
def _render_processed_history() -> None:
    """Show a summary of all files processed in this session."""
    processed = st.session_state.get(_KEY_PROCESSED_FILES, {})
    if not processed:
        return
 
    st.markdown("---")
    st.subheader("📁 Processed in This Session")
 
    for file_name, result in processed.items():
        status_icon = "✅" if not result.get("error") else "❌"
        with st.expander(f"{status_icon} {file_name}"):
            _display_results(result)
 
