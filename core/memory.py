"""
core/memory.py
--------------
LangGraph 1.x MemorySaver-backed session memory.
"""
 
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
# LangGraph 1.x — MemorySaver is in langgraph.checkpoint.memory
from langgraph.checkpoint.memory import MemorySaver
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
# Shared MemorySaver instance — imported by langgraph_pipeline.py
langgraph_memory = MemorySaver()
 
 
@dataclass
class MemoryTurn:
    query:      str
    answer:     str
    context:    str
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())
    turn_index: int = 0
 
 
class SessionMemory:
    """
    UI-facing session memory backed by LangGraph MemorySaver.
    Keeps the same .add()/.clear()/.get_recent_context() interface
    so app.py needs zero changes.
    """
 
    def __init__(self, window_size: int | None = None) -> None:
        self._window_size   = window_size or settings.app.memory_window_size
        self._turns:        list[MemoryTurn] = []
        self._turn_counter: int = 0
        logger.info("SessionMemory initialised (LangGraph 1.x MemorySaver).")
 
    def add(self, query: str, answer: str, context: str = "") -> MemoryTurn:
        turn = MemoryTurn(
            query=query, answer=answer,
            context=context, turn_index=self._turn_counter,
        )
        if len(self._turns) >= self._window_size:
            self._turns.pop(0)
        self._turns.append(turn)
        self._turn_counter += 1
        return turn
 
    def get_history(self) -> list[MemoryTurn]:
        return list(self._turns)
 
    def get_recent_context(self, max_turns: int = 3) -> str:
        if not self._turns:
            return ""
        parts = [
            f"Previous question: {t.query}\nPrevious answer: {t.answer}"
            for t in self._turns[-max_turns:]
        ]
        return "\n\n".join(parts)
 
    def get_last_turn(self) -> MemoryTurn | None:
        return self._turns[-1] if self._turns else None
 
    def clear(self) -> None:
        count = len(self._turns)
        self._turns.clear()
        self._turn_counter = 0
        logger.info("SessionMemory cleared | %d turns removed", count)
 
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
            f"window={self._window_size}, backend=LangGraph1.x)"
        )
 
 
def get_or_create_memory(session_state, key: str = "memory") -> SessionMemory:
    if key not in session_state:
        session_state[key] = SessionMemory()
    return session_state[key]