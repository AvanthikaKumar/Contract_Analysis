"""
ui/tab_query.py
----------------
Streamlit Tab 2 — Contract Intelligence (Query Interface).
 
Responsibilities:
- Accept natural language queries from the user.
- Call the QueryEngine to retrieve and generate answers.
- Display answers, supporting context, and conversation history.
- Maintain session memory across follow-up questions.
- Support contract summarisation as a one-click action.
 
This module contains only UI logic. All business logic is
delegated to the retrieval and llm modules.
"""
 
import logging
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
import streamlit as st
 
from core.memory import get_or_create_memory
from ingestion.embedder import embedder
from ingestion.vector_store import vector_store
from retrieval.query_engine import query_engine
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
_KEY_CHAT_HISTORY    = "chat_history"
_KEY_PROCESSED_FILES = "processed_files"
 
 
# ---------------------------------------------------------------------------
# Helper — check if any contracts are indexed
# ---------------------------------------------------------------------------
def _has_indexed_contracts() -> bool:
    """Return True if the vector index has at least one document."""
    try:
        count = vector_store.get_document_count()
        return count > 0
    except Exception:
        return False
 
 
# ---------------------------------------------------------------------------
# Chat history management
# ---------------------------------------------------------------------------
def _get_chat_history() -> list[dict]:
    if _KEY_CHAT_HISTORY not in st.session_state:
        st.session_state[_KEY_CHAT_HISTORY] = []
    return st.session_state[_KEY_CHAT_HISTORY]
 
 
def _add_to_chat(role: str, content: str, metadata: dict | None = None) -> None:
    """Append a message to the chat history."""
    _get_chat_history().append({
        "role":     role,
        "content":  content,
        "metadata": metadata or {},
    })
 
 
def _render_chat_history() -> None:
    """Render all previous messages in the chat."""
    for message in _get_chat_history():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
            # Show supporting context in an expander for assistant messages
            meta = message.get("metadata", {})
            if message["role"] == "assistant" and meta.get("sources"):
                with st.expander("📄 Supporting Contract Context", expanded=False):
                    for i, source in enumerate(meta["sources"], start=1):
                        st.markdown(f"**Excerpt {i}**")
                        st.text(source[:600] + ("..." if len(source) > 600 else ""))
                        if i < len(meta["sources"]):
                            st.divider()
 
 
# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------
def _handle_query(query: str) -> None:
    """Process a user query and render the response."""
    memory = get_or_create_memory(st.session_state)
 
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    _add_to_chat("user", query)
 
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.run(
                query=query,
                memory=memory,
            )
 
        # Out of scope
        if response.is_out_of_scope:
            st.warning(response.answer)
            _add_to_chat("assistant", response.answer)
            return
 
        # Grounded answer
        st.markdown(response.answer)
 
        # Show sources
        if response.sources:
            with st.expander("📄 Supporting Contract Context", expanded=False):
                for i, source in enumerate(response.sources, start=1):
                    st.markdown(f"**Excerpt {i}**")
                    st.text(source[:600] + ("..." if len(source) > 600 else ""))
                    if i < len(response.sources):
                        st.divider()
 
        # Show source files
        if response.source_files:
            st.caption(
                "📁 Sources: " + ", ".join(f"`{f}`" for f in response.source_files)
            )
 
        # Not grounded warning
        if not response.is_grounded:
            st.info("ℹ️ The answer could not be found in the contract context.")
 
    _add_to_chat(
        "assistant",
        response.answer,
        metadata={"sources": response.sources, "source_files": response.source_files},
    )
 
 
# ---------------------------------------------------------------------------
# Summarise handler
# ---------------------------------------------------------------------------
def _handle_summarise(source_file: str | None = None) -> None:
    """Generate and display a contract summary."""
    memory = get_or_create_memory(st.session_state)
 
    with st.chat_message("user"):
        st.markdown("📋 Summarise the contract.")
    _add_to_chat("user", "Summarise the contract.")
 
    with st.chat_message("assistant"):
        with st.spinner("Generating summary..."):
            # Retrieve broad context for summary
            query_vector = embedder.embed_query(
                "contract summary parties obligations payment termination"
            )
            results = vector_store.search(
                query_vector=query_vector,
                top_k=10,
                source_file=source_file,
            )
            combined = "\n\n".join(r.text for r in results)
            response = query_engine.summarise(combined, memory=memory)
 
        st.markdown(response.answer)
 
    _add_to_chat("assistant", response.answer)
 
 
# ---------------------------------------------------------------------------
# Main tab render function
# ---------------------------------------------------------------------------
def render_query_tab() -> None:
    """Render the full Contract Intelligence query tab."""
 
    st.header("🔍 Contract Intelligence")
    st.markdown(
        "Ask natural language questions about your uploaded contracts. "
        "The system retrieves relevant clauses and generates grounded answers."
    )
 
    # -------------------------------------------------------------------
    # Guard — check contracts are indexed
    # -------------------------------------------------------------------
    if not _has_indexed_contracts():
        st.warning(
            "⚠️ No contracts indexed yet. "
            "Please upload and process a contract in the **Upload** tab first."
        )
        return
 
    # -------------------------------------------------------------------
    # Sidebar — session controls
    # -------------------------------------------------------------------
    with st.sidebar:
        st.subheader("⚙️ Session Controls")
 
        # Source file filter
        processed = st.session_state.get(_KEY_PROCESSED_FILES, {})
        file_options = ["All contracts"] + [
            name for name, r in processed.items() if not r.get("error")
        ]
 
        selected_file = st.selectbox(
            "Filter by contract",
            options=file_options,
            help="Restrict answers to a specific contract document.",
        )
        source_filter = (
            None if selected_file == "All contracts" else selected_file
        )
 
        st.divider()
 
        # Summarise button
        if st.button("📋 Summarise Contract", use_container_width=True):
            _handle_summarise(source_file=source_filter)
 
        st.divider()
 
        # Example queries
        st.markdown("**💡 Example queries:**")
        example_queries = [
            "Who are the parties in this agreement?",
            "What is the contract start date?",
            "What are the payment terms?",
            "What are the termination conditions?",
            "What is the governing law?",
            "Explain the pricing adjustment clauses.",
        ]
        for example in example_queries:
            if st.button(
                example,
                use_container_width=True,
                key=f"example_{example[:20]}",
            ):
                _handle_query(example)
 
        st.divider()
 
        # Memory stats
        memory = get_or_create_memory(st.session_state)
        st.caption(
            f"🧠 Memory: {memory.turn_count} turns in window "
            f"({memory.total_turns} total this session)"
        )
 
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state[_KEY_CHAT_HISTORY] = []
            memory.clear()
            st.rerun()
 
    # -------------------------------------------------------------------
    # Chat history
    # -------------------------------------------------------------------
    _render_chat_history()
 
    # -------------------------------------------------------------------
    # Chat input
    # -------------------------------------------------------------------
    query = st.chat_input(
        placeholder="Ask a question about your contract...",
    )
 
    if query:
        _handle_query(query)
 