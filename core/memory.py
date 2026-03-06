"""
core/memory.py
--------------
Session-level short-term memory for the Contract Intelligence System.
 
Responsibilities:
- Store recent user queries, answers, and retrieved context per session.
- Provide recent conversation history for follow-up query enrichment.
- Enforce a sliding window — oldest turns are dropped when limit is reached.
- Never persist data between sessions (session-scoped only).
- Integrate cleanly with Streamlit's st.session_state.
 
Usage:
    from core.memory import SessionMemory
 
    memory = SessionMemory()
    memory.add(query="What are the payment terms?", answer="...", context="...")
 
    # Get recent context for query enrichment
    recent = memory.get_recent_context()
 
    # Get full history for display in UI
    history = memory.get_history()
"""
 
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Memory turn dataclass
# ---------------------------------------------------------------------------
@dataclass
class MemoryTurn:
    """
    A single conversation turn stored in session memory.
 
    Attributes:
        query:      The user's question.
        answer:     The generated answer.
        context:    The retrieved contract context used for the answer.
        timestamp:  When this turn was created.
        turn_index: Position in the conversation (0-based).
    """
    query: str
    answer: str
    context: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_index: int = 0
 
    def __repr__(self) -> str:
        return (
            f"MemoryTurn("
            f"turn={self.turn_index}, "
            f"query='{self.query[:50]}', "
            f"answer_chars={len(self.answer)})"
        )
 
 
# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------
class SessionMemory:
    """
    Sliding window short-term memory scoped to a single user session.
 
    Stores the last N conversation turns. When the window is full,
    the oldest turn is dropped to make room for the newest.
 
    Designed to work with Streamlit's st.session_state — the caller
    stores the SessionMemory instance in session state and passes it
    to the QueryEngine on each run.
 
    Attributes:
        _turns:       Ordered list of MemoryTurn objects (oldest first).
        _window_size: Maximum number of turns to retain.
        _turn_counter: Running count of total turns added this session.
    """
 
    def __init__(self, window_size: int | None = None) -> None:
        self._window_size = window_size or settings.app.memory_window_size
        self._turns: list[MemoryTurn] = []
        self._turn_counter: int = 0
        logger.info(
            "SessionMemory initialised | window_size=%d", self._window_size
        )
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(
        self,
        query: str,
        answer: str,
        context: str = "",
    ) -> MemoryTurn:
        """
        Add a new conversation turn to memory.
 
        If the memory window is full, the oldest turn is evicted
        before the new one is added.
 
        Args:
            query:   The user's question.
            answer:  The generated answer.
            context: The retrieved context used (optional).
 
        Returns:
            The newly created MemoryTurn.
        """
        turn = MemoryTurn(
            query=query,
            answer=answer,
            context=context,
            turn_index=self._turn_counter,
        )
 
        # Enforce sliding window
        if len(self._turns) >= self._window_size:
            evicted = self._turns.pop(0)
            logger.debug(
                "Evicted oldest memory turn: turn=%d", evicted.turn_index
            )
 
        self._turns.append(turn)
        self._turn_counter += 1
 
        logger.debug(
            "Memory turn added | turn=%d | window=%d/%d",
            turn.turn_index,
            len(self._turns),
            self._window_size,
        )
 
        return turn
 
    def get_history(self) -> list[MemoryTurn]:
        """
        Return all turns currently in the memory window.
 
        Returns:
            List of MemoryTurn objects, oldest first.
        """
        return list(self._turns)
 
    def get_recent_context(self, max_turns: int = 3) -> str:
        """
        Build a concise recent conversation summary for query enrichment.
 
        This is injected into the query before embedding to improve
        retrieval accuracy for follow-up questions.
 
        Args:
            max_turns: Maximum number of recent turns to include.
 
        Returns:
            Formatted string of recent Q&A pairs, or empty string if
            no history exists.
        """
        if not self._turns:
            return ""
 
        recent = self._turns[-max_turns:]
        parts = []
 
        for turn in recent:
            parts.append(
                f"Previous question: {turn.query}\n"
                f"Previous answer: {turn.answer}"
            )
 
        return "\n\n".join(parts)
 
    def get_last_turn(self) -> MemoryTurn | None:
        """
        Return the most recent memory turn.
 
        Returns:
            The latest MemoryTurn, or None if memory is empty.
        """
        if not self._turns:
            return None
        return self._turns[-1]
 
    def clear(self) -> None:
        """
        Clear all turns from memory.
        Used when the user starts a new conversation or uploads a new contract.
        """
        count = len(self._turns)
        self._turns.clear()
        self._turn_counter = 0
        logger.info("SessionMemory cleared | removed=%d turns", count)
 
    def is_empty(self) -> bool:
        """Return True if no turns are stored."""
        return len(self._turns) == 0
 
    @property
    def turn_count(self) -> int:
        """Return the number of turns currently in the window."""
        return len(self._turns)
 
    @property
    def total_turns(self) -> int:
        """Return the total number of turns added this session (including evicted)."""
        return self._turn_counter
 
    def __len__(self) -> int:
        return len(self._turns)
 
    def __repr__(self) -> str:
        return (
            f"SessionMemory("
            f"turns={len(self._turns)}, "
            f"window={self._window_size}, "
            f"total_added={self._turn_counter})"
        )
 
 
# ---------------------------------------------------------------------------
# Streamlit integration helper
# ---------------------------------------------------------------------------
def get_or_create_memory(session_state, key: str = "memory") -> SessionMemory:
    """
    Get or create a SessionMemory instance from Streamlit's session_state.
 
    This is a convenience function for use in the UI layer. It ensures
    a single SessionMemory instance persists across Streamlit reruns
    within the same session.
 
    Args:
        session_state: Streamlit's st.session_state object.
        key:           The key to use in session_state.
 
    Returns:
        The existing or newly created SessionMemory instance.
 
    Usage (in Streamlit):
        from core.memory import get_or_create_memory
        memory = get_or_create_memory(st.session_state)
    """
    if key not in session_state:
        session_state[key] = SessionMemory()
        logger.info("Created new SessionMemory in session_state['%s'].", key)
    return session_state[key]
 
 
# ---------------------------------------------------------------------------
# Update core/__init__.py exports
# ---------------------------------------------------------------------------
 
 
# ---------------------------------------------------------------------------
# Smoke test — python core/memory.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== SessionMemory Smoke Test ===\n")
 
    mem = SessionMemory(window_size=3)
 
    print(f"Initial state: {mem}")
    print(f"Is empty: {mem.is_empty()}\n")
 
    # Add turns
    turns_data = [
        (
            "Who are the parties in this agreement?",
            "The parties are Acme Corporation (Client) and TechVendor Inc. (Vendor).",
            "Acme Corporation and TechVendor Inc. entered into this agreement...",
        ),
        (
            "What are the payment terms?",
            "Client shall pay Vendor $50,000 monthly, due within 30 days of invoice.",
            "Payment is due within 30 days. Late payments accrue 1.5% monthly interest.",
        ),
        (
            "What is the termination notice period?",
            "Either party may terminate with 30 days written notice.",
            "Either party may terminate this Agreement with 30 days written notice.",
        ),
        (
            "What happens to outstanding invoices on termination?",
            "Client shall pay all outstanding invoices within 15 days of termination.",
            "Upon termination, Client shall pay all outstanding invoices within 15 days.",
        ),
    ]
 
    print("--- Adding 4 turns (window size = 3) ---\n")
    for query, answer, context in turns_data:
        mem.add(query=query, answer=answer, context=context)
        print(f"  After add: {mem}")
 
    print(f"\nTotal turns added : {mem.total_turns}")
    print(f"Turns in window   : {mem.turn_count}")
 
    print("\n--- Current history in window ---\n")
    for turn in mem.get_history():
        print(f"  {turn}")
 
    print("\n--- Recent context (for query enrichment) ---\n")
    recent = mem.get_recent_context(max_turns=2)
    print(recent)
 
    print("\n--- Last turn ---")
    last = mem.get_last_turn()
    print(f"  {last}")
 
    print("\n--- Clearing memory ---")
    mem.clear()
    print(f"  After clear: {mem}")
    print(f"  Is empty: {mem.is_empty()}")
 
    print("\n=== Smoke test complete ===\n")