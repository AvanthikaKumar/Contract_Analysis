"""
ingestion/document_loader.py
-----------------------------
Extracts raw text from uploaded contract PDF files using
Azure Document Intelligence (formerly Form Recognizer).
 
Responsibilities:
- Accept raw PDF bytes from Streamlit file uploader.
- Send bytes to Azure Document Intelligence prebuilt-read model.
- Return clean extracted text with page structure preserved.
- Handle multi-page documents gracefully.
 
Usage:
    from ingestion.document_loader import document_loader
 
    text = document_loader.extract_text(file_bytes, file_name)
"""
 
import logging
import sys
from pathlib import Path
from typing import Optional
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
_RETRY_POLICY = dict(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
 
 
# ---------------------------------------------------------------------------
# Extracted document dataclass
# ---------------------------------------------------------------------------
class ExtractedDocument:
    """
    Holds the result of a document extraction.
 
    Attributes:
        file_name:   Original uploaded file name.
        full_text:   Complete extracted text from all pages.
        pages:       List of per-page text strings.
        page_count:  Total number of pages detected.
    """
 
    def __init__(
        self,
        file_name: str,
        full_text: str,
        pages: list[str],
        page_count: int,
    ) -> None:
        self.file_name = file_name
        self.full_text = full_text
        self.pages = pages
        self.page_count = page_count
 
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
    Wraps Azure Document Intelligence to extract text from PDF bytes.
 
    Uses the prebuilt-read model which is optimised for dense text
    documents like contracts and legal agreements.
    """
 
    def __init__(self) -> None:
        cfg = settings.document_intelligence
        self._client = DocumentIntelligenceClient(
            endpoint=cfg.endpoint,
            credential=AzureKeyCredential(cfg.api_key),
        )
        logger.info(
            "DocumentLoader initialised | endpoint=%s", cfg.endpoint
        )
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @retry(**_RETRY_POLICY)
    def extract_text(
        self,
        file_bytes: bytes,
        file_name: str = "document.pdf",
    ) -> ExtractedDocument:
        """
        Extract text from a PDF supplied as raw bytes.
 
        Args:
            file_bytes: Raw bytes of the uploaded PDF file.
            file_name:  Original filename (used for logging and metadata).
 
        Returns:
            ExtractedDocument with full text and per-page breakdown.
 
        Raises:
            ValueError: If file_bytes is empty.
            Exception:  If Azure Document Intelligence call fails after retries.
        """
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty.")
 
        logger.info(
            "Starting text extraction | file=%s | size=%.1f KB",
            file_name,
            len(file_bytes) / 1024,
        )
 
        # Submit to Azure Document Intelligence
        # SDK v1.0.0 requires model_id and body as positional arguments
        poller = self._client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=file_bytes),
            output_content_format="markdown",  # better structure preservation
        )
 
        result = poller.result()
 
        # Build per-page text list — skip blank pages but continue processing
        pages: list[str] = []
        blank_pages: list[int] = []
        page_position = 0
 
        for page in result.pages or []:
            page_position += 1
            page_lines = []
            for line in page.lines or []:
                line_text = line.content.strip()
                if line_text:
                    page_lines.append(line_text)
 
            page_text = "\n".join(page_lines).strip()
 
            if not page_text:
                blank_pages.append(page_position)
                logger.info(
                    "Blank page at position %d — skipping, continuing extraction.",
                    page_position,
                )
                continue
 
            pages.append(page_text)
 
        if blank_pages:
            logger.info(
                "Skipped %d blank page(s) at positions: %s",
                len(blank_pages),
                blank_pages,
            )
 
        full_text = "\n\n".join(pages)
        page_count = len(pages)
 
        logger.info(
            "Extraction complete | file=%s | pages=%d | chars=%d",
            file_name,
            page_count,
            len(full_text),
        )
 
        return ExtractedDocument(
            file_name=file_name,
            full_text=full_text,
            pages=pages,
            page_count=page_count,
        )
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
document_loader = DocumentLoader()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python ingestion/document_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
 
    print("\n=== DocumentLoader Smoke Test ===\n")
 
    # Find all PDFs in project root
    pdf_files = sorted(_PROJECT_ROOT.glob("*.pdf"))
 
    if not pdf_files:
        print("No PDFs found in project root for testing.")
        print("Place any PDF in the project root and re-run.")
        sys.exit(0)
 
    print(f"Found {len(pdf_files)} PDF(s):\n")
    for f in pdf_files:
        print(f"  - {f.name}")
    print()
 
    loader = DocumentLoader()
 
    for test_pdf in pdf_files:
        print(f"--- Processing: {test_pdf.name} ---")
        try:
            with open(test_pdf, "rb") as fh:
                raw_bytes = fh.read()
 
            doc = loader.extract_text(raw_bytes, test_pdf.name)
 
            print(f"  File Name  : {doc.file_name}")
            print(f"  Pages      : {doc.page_count}")
            print(f"  Total chars: {len(doc.full_text)}")
            print(f"  First 300 chars:")
            print(f"  {doc.full_text[:300]}")
            print(f"  PASSED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")
 