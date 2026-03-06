"""
ingestion/chunker.py
---------------------
Splits extracted contract text into overlapping chunks for embedding.
 
Responsibilities:
- Split long contract text into fixed-size chunks with overlap.
- Preserve sentence boundaries where possible to avoid mid-sentence cuts.
- Attach metadata (chunk index, char offset, source file) to each chunk.
- Return a clean list of Chunk objects ready for embedding.
 
Usage:
    from ingestion.chunker import chunker
 
    chunks = chunker.split(extracted_document)
"""
 
import logging
import sys
from dataclasses import dataclass, field
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
# Chunk dataclass
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage.
 
    Attributes:
        chunk_id:     Unique identifier: "{file_stem}_chunk_{index}"
        text:         The chunk text content.
        index:        Zero-based position in the chunk sequence.
        char_start:   Character offset in the original full text.
        char_end:     End character offset in the original full text.
        source_file:  Original filename the chunk came from.
        metadata:     Additional key-value metadata for storage.
    """
    chunk_id: str
    text: str
    index: int
    char_start: int
    char_end: int
    source_file: str
    metadata: dict = field(default_factory=dict)
 
    def __repr__(self) -> str:
        return (
            f"Chunk(id='{self.chunk_id}', "
            f"index={self.index}, "
            f"chars={len(self.text)})"
        )
 
 
# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class Chunker:
    """
    Splits contract text into overlapping chunks with sentence awareness.
 
    Uses character-based chunking with overlap, but respects sentence
    boundaries by finding the nearest sentence end within a tolerance
    window — avoiding cuts in the middle of a sentence.
 
    Attributes:
        chunk_size:    Target chunk size in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        tolerance:     Characters to look ahead/behind for sentence boundary.
    """
 
    # Sentence-ending punctuation to break on
    _SENTENCE_ENDS = {".", "!", "?", "\n\n"}
 
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        tolerance: int = 100,
    ) -> None:
        self.chunk_size = chunk_size or settings.app.chunk_size
        self.chunk_overlap = chunk_overlap or settings.app.chunk_overlap
        self.tolerance = tolerance
 
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})."
            )
 
        logger.info(
            "Chunker initialised | size=%d | overlap=%d | tolerance=%d",
            self.chunk_size,
            self.chunk_overlap,
            self.tolerance,
        )
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(
        self,
        extracted_document,
    ) -> list[Chunk]:
        """
        Split an ExtractedDocument into a list of Chunk objects.
 
        Args:
            extracted_document: An ExtractedDocument from DocumentLoader.
 
        Returns:
            List of Chunk objects ordered by position in the document.
        """
        return self.split_text(
            text=extracted_document.full_text,
            source_file=extracted_document.file_name,
        )
 
    def split_text(
        self,
        text: str,
        source_file: str = "unknown",
    ) -> list[Chunk]:
        """
        Split a raw text string into overlapping Chunk objects.
 
        Args:
            text:        The full contract text to split.
            source_file: Name of the source file for metadata.
 
        Returns:
            List of Chunk objects.
 
        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text.")
 
        text = text.strip()
        file_stem = Path(source_file).stem
        safe_stem = self._sanitise_key(file_stem)
 
        chunks: list[Chunk] = []
        start = 0
        index = 0
 
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
 
            # Snap end to nearest sentence boundary within tolerance
            if end < len(text):
                end = self._find_sentence_boundary(text, end)
 
            chunk_text = text[start:end].strip()
 
            if chunk_text:
                chunk = Chunk(
                    chunk_id=f"{safe_stem}_chunk_{index:04d}",
                    text=chunk_text,
                    index=index,
                    char_start=start,
                    char_end=end,
                    source_file=source_file,
                    metadata={
                        "source_file": source_file,
                        "chunk_index": index,
                        "char_start": start,
                        "char_end": end,
                        "chunk_size": len(chunk_text),
                    },
                )
                chunks.append(chunk)
                logger.debug(
                    "Created %s | chars=%d", chunk.chunk_id, len(chunk_text)
                )
                index += 1
 
            # Advance start with overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1  # Guard against infinite loop
            start = next_start
 
        logger.info(
            "Chunking complete | file=%s | chunks=%d | avg_chars=%.0f",
            source_file,
            len(chunks),
            sum(len(c.text) for c in chunks) / max(len(chunks), 1),
        )
 
        return chunks
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise_key(name: str) -> str:
        """
        Sanitise a string so it is safe to use as an Azure AI Search document key.
        Allowed characters: letters, digits, underscore, dash, equal sign.
        All other characters are replaced with underscores.
        """
        import re
        # Remove file extension
        stem = Path(name).stem
        # Replace any character that is not alphanumeric, dash, or underscore
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)
        # Collapse multiple underscores
        safe = re.sub(r'_+', '_', safe)
        # Strip leading/trailing underscores
        safe = safe.strip('_')
        # Truncate to 100 chars to keep keys manageable
        return safe[:100]
 
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary around the given position.
 
        Searches within [position - tolerance, position + tolerance]
        for a sentence-ending character. Falls back to position if
        no boundary is found.
 
        Args:
            text:     Full text string.
            position: Target split position.
 
        Returns:
            Adjusted position at or near a sentence boundary.
        """
        search_start = max(0, position - self.tolerance)
        search_end = min(len(text), position + self.tolerance)
        window = text[search_start:search_end]
 
        # Search backwards from the target for a sentence end
        best_pos = position
        for i in range(len(window) - 1, -1, -1):
            char = window[i]
            if char in {".", "!", "?"}:
                # Include the punctuation character in this chunk
                best_pos = search_start + i + 1
                break
            if char == "\n" and i > 0 and window[i - 1] == "\n":
                best_pos = search_start + i + 1
                break
 
        return best_pos
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
chunker = Chunker()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python ingestion/chunker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Chunker Smoke Test ===\n")
 
    sample_text = """
    THIS MASTER SERVICES AGREEMENT ("Agreement") is entered into as of January 1, 2025,
    by and between Acme Corporation, a Delaware corporation ("Client"), and TechVendor Inc.,
    a California corporation ("Vendor").
 
    1. SERVICES
    Vendor agrees to provide software development and consulting services as described
    in the Statement of Work attached hereto as Exhibit A. Vendor shall commence services
    on February 1, 2025, and shall complete all deliverables by December 31, 2025.
 
    2. PAYMENT TERMS
    Client shall pay Vendor a monthly retainer of $50,000, due within 30 days of invoice.
    Late payments shall accrue interest at 1.5% per month. In the event of non-payment,
    Vendor reserves the right to suspend services with 7 days written notice.
 
    3. TERMINATION
    Either party may terminate this Agreement with 30 days written notice. Upon termination,
    Client shall pay all outstanding invoices within 15 days. Vendor shall deliver all
    work product completed as of the termination date.
 
    4. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws of
    the State of Delaware, without regard to its conflict of law provisions.
    """ * 5  # Repeat to simulate a longer document
 
    c = Chunker(chunk_size=500, chunk_overlap=100)
    chunks = c.split_text(sample_text.strip(), source_file="test_contract.pdf")
 
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Chunk size      : {c.chunk_size}")
    print(f"  Chunk overlap   : {c.chunk_overlap}")
    print()
 
    for chunk in chunks[:3]:
        print(f"  --- {chunk.chunk_id} ---")
        print(f"  Chars     : {len(chunk.text)}")
        print(f"  Start/End : {chunk.char_start} / {chunk.char_end}")
        print(f"  Preview   : {chunk.text[:120]}...")
        print()
 
    print("=== Smoke test complete ===\n")