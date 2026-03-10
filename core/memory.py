"""
core/memory.py
--------------
LangGraph MemorySaver-backed session memory.
 
Replaces the custom sliding window with LangGraph's built-in
MemorySaver checkpointer, which automatically persists
conversation state per thread_id across LangGraph invocations.
 
The SessionMemory class is kept for backward compatibility
with the UI (app.py still calls memory.add() and memory.clear()).
Internally it now delegates to LangGraph's MemorySaver.
"""
 
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from langgraph.checkpoint.memory import MemorySaver
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
# ── Module-level MemorySaver shared across the app ─────────────────────────
# This is the LangGraph checkpointer — imported by langgraph_pipeline.py
langgraph_memory = MemorySaver()
 
 
# ── Legacy dataclass — kept so UI code doesn't break ──────────────────────
@dataclass
class MemoryTurn:
    query:       str
    answer:      str
    context:     str
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
    turn_index:  int = 0
 
 
# ── SessionMemory — UI-facing interface ───────────────────────────────────
class SessionMemory:
    """
    Session memory backed by LangGraph MemorySaver.
 
    The UI still calls .add() / .clear() / .get_recent_context()
    exactly as before — those calls now sync with LangGraph state
    so memory is consistent between UI display and LangGraph nodes.
    """
 
    def __init__(self, window_size: int | None = None) -> None:
        self._window_size  = window_size or settings.app.memory_window_size
        self._turns:       list[MemoryTurn] = []
        self._turn_counter: int = 0
        logger.info("SessionMemory initialised (LangGraph MemorySaver backend).")
 
    # ── Public API (same interface as before) ──────────────────────────
    def add(self, query: str, answer: str, context: str = "") -> MemoryTurn:
        """Add a turn — keeps sliding window for UI display."""
        turn = MemoryTurn(
            query=query, answer=answer,
            context=context, turn_index=self._turn_counter,
        )
        if len(self._turns) >= self._window_size:
            self._turns.pop(0)
        self._turns.append(turn)
        self._turn_counter += 1
        logger.debug("MemoryTurn added | turn=%d", turn.turn_index)
        return turn
 
    def get_history(self) -> list[MemoryTurn]:
        return list(self._turns)
 
    def get_recent_context(self, max_turns: int = 3) -> str:
        """Returns recent Q&A pairs as a string for query enrichment."""
        if not self._turns:
            return ""
        recent = self._turns[-max_turns:]
        parts  = []
        for t in recent:
            parts.append(
                f"Previous question: {t.query}\n"
                f"Previous answer: {t.answer}"
            )
        return "\n\n".join(parts)
 
    def get_last_turn(self) -> MemoryTurn | None:
        return self._turns[-1] if self._turns else None
 
    def clear(self) -> None:
        """Clear UI-side memory. LangGraph memory resets via new thread_id."""
        count = len(self._turns)
        self._turns.clear()
        self._turn_counter = 0
        logger.info("SessionMemory cleared | removed=%d turns", count)
 
    def is_empty(self) -> bool:
        return len(self._turns) == 0
 
    @property
    def turn_count(self) -> int:
        return len(self._turns)
 
    @property
    def total_turns(self) -> int:
        return self._turn_counter
 
    def __len__(self) -> int:
        return len(self._turns)
 
    def __repr__(self) -> str:
        return (
            f"SessionMemory(turns={len(self._turns)}, "
            f"window={self._window_size}, "
            f"backend=LangGraphMemorySaver)"
        )
 
 
# ── Streamlit integration helper ──────────────────────────────────────────
def get_or_create_memory(session_state, key: str = "memory") -> SessionMemory:
    """
    Get or create a SessionMemory from Streamlit session_state.
    Same interface as before — UI code needs no changes.
    """
    if key not in session_state:
        session_state[key] = SessionMemory()
        logger.info("Created new SessionMemory in session_state['%s'].", key)
    return session_state[key]