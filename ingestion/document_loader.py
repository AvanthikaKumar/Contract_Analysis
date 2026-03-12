"""
ingestion/document_loader.py
-----------------------------
Extracts text from PDF contracts using PyPDF — free, local, no API limits.
 
Replaces Azure Document Intelligence (which has a 2-page free tier limit).
PyPDF reads all pages regardless of document length.
 
Responsibilities:
- Accept raw PDF bytes from Streamlit file uploader.
- Extract text from every page using PyPDF.
- Skip blank pages and continue processing.
- Return a clean ExtractedDocument ready for chunking.
 
Usage:
    from ingestion.document_loader import document_loader
 
    doc = document_loader.extract_text(file_bytes, file_name)
    print(doc.page_count)
    print(doc.full_text[:500])
"""
 
import io
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
 
from pypdf import PdfReader
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Extracted document dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractedDocument:
    """
    The result of text extraction from a PDF.
 
    Attributes:
        file_name:  Original filename.
        full_text:  All extracted text joined across pages.
        pages:      List of per-page text strings (blank pages excluded).
        page_count: Number of non-blank pages successfully extracted.
    """
    file_name: str
    full_text: str
    pages: list[str] = field(default_factory=list)
    page_count: int = 0
 
    def __repr__(self) -> str:
        return (
            f"ExtractedDocument("
            f"file='{self.file_name}', "
            f"pages={self.page_count}, "
            f"chars={len(self.full_text)})"
        )
 
 
# ---------------------------------------------------------------------------
# Document loader
# ---------------------------------------------------------------------------
class DocumentLoader:
    """
    Extracts text from PDF files using PyPDF.
 
    Processes all pages in the document with no page limit.
    Blank pages are logged and skipped — extraction continues
    through the rest of the document.
    """
 
    def __init__(self) -> None:
        logger.info("DocumentLoader initialised (PyPDF backend).")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_text(
        self,
        file_bytes: bytes,
        file_name: str,
    ) -> ExtractedDocument:
        """
        Extract text from a PDF supplied as raw bytes.
 
        Args:
            file_bytes: Raw PDF bytes from Streamlit file uploader.
            file_name:  Original filename for metadata.
 
        Returns:
            ExtractedDocument with full text and per-page breakdown.
 
        Raises:
            ValueError: If file_bytes is empty.
            RuntimeError: If the PDF cannot be read.
        """
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty.")
 
        logger.info("Extracting text from '%s' (%d bytes)", file_name, len(file_bytes))
 
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open PDF '{file_name}': {exc}"
            ) from exc
 
        total_pages = len(reader.pages)
        logger.info("PDF has %d total pages.", total_pages)
 
        pages: list[str] = []
        blank_pages: list[int] = []
 
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
                page_text = raw_text.strip()
            except Exception as exc:
                logger.warning("Failed to extract text from page %d: %s", i, exc)
                page_text = ""
 
            if not page_text:
                blank_pages.append(i)
                logger.debug("Page %d is blank — skipping, continuing.", i)
                continue
 
            pages.append(page_text)
 
        if blank_pages:
            logger.info(
                "Skipped %d blank page(s): %s", len(blank_pages), blank_pages
            )
 
        full_text = "\n\n".join(pages)
 
        logger.info(
            "Extraction complete | file=%s | total_pages=%d | "
            "extracted=%d | blank=%d | chars=%d",
            file_name,
            total_pages,
            len(pages),
            len(blank_pages),
            len(full_text),
        )
 
        return ExtractedDocument(
            file_name=file_name,
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
        )
 
    def extract_from_path(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Convenience method to extract text directly from a file path.
 
        Args:
            pdf_path: Path to the PDF file.
 
        Returns:
            ExtractedDocument with extracted text.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
 
        file_bytes = pdf_path.read_bytes()
        return self.extract_text(file_bytes, pdf_path.name)
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
document_loader = DocumentLoader()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python ingestion/document_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== DocumentLoader Smoke Test (PyPDF) ===\n")
 
    # Find PDFs in project root
    pdf_files = sorted(_PROJECT_ROOT.glob("*.pdf"))
 
    if not pdf_files:
        print("No PDF files found in project root.")
        print("Place a PDF in the project root and re-run.\n")
        sys.exit(0)
 
    loader = DocumentLoader()
 
    for pdf_path in pdf_files:
        print(f"Testing: {pdf_path.name}")
        try:
            doc = loader.extract_from_path(pdf_path)
            print(f"  Total pages extracted : {doc.page_count}")
            print(f"  Total characters      : {len(doc.full_text):,}")
            print(f"  Preview (first 300)   : {doc.full_text[:300].strip()}")
            print(f"  PASSED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")