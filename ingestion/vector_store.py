"""
ingestion/vector_store.py
--------------------------
Stores and retrieves contract chunk embeddings using Azure AI Search.
 
Responsibilities:
- Create the Azure AI Search vector index if it does not exist.
- Upload embedded chunks (text + vector + metadata) to the index.
- Perform semantic vector search given a query embedding.
- Return top-K relevant chunks for LLM context assembly.
 
Usage:
    from ingestion.vector_store import vector_store
 
    # During ingestion
    vector_store.upload_chunks(embedded_chunks)
 
    # During retrieval
    results = vector_store.search(query_vector, top_k=5)
"""
 
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
 
from config.settings import settings
from ingestion.embedder import EmbeddedChunk
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Index schema constants
# ---------------------------------------------------------------------------
_FIELD_ID = "chunk_id"
_FIELD_TEXT = "text"
_FIELD_SOURCE = "source_file"
_FIELD_INDEX = "chunk_index"
_FIELD_VECTOR = "embedding"
_VECTOR_DIMENSIONS = 1536       # text-embedding-3-large output size
_ALGORITHM_NAME = "hnsw-config"
_PROFILE_NAME = "vector-profile"
 
 
# ---------------------------------------------------------------------------
# Search result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """
    A single retrieved chunk from Azure AI Search.
 
    Attributes:
        chunk_id:    Unique chunk identifier.
        text:        The chunk text content.
        source_file: Original contract filename.
        score:       Relevance score from the search engine.
    """
    chunk_id: str
    text: str
    source_file: str
    score: float
 
    def __repr__(self) -> str:
        return (
            f"SearchResult(id='{self.chunk_id}', "
            f"score={self.score:.4f}, "
            f"chars={len(self.text)})"
        )
 
 
# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class VectorStore:
    """
    Manages Azure AI Search index creation, upload, and vector retrieval.
 
    The index is created automatically on first use if it does not exist.
    Uses HNSW approximate nearest-neighbour search for fast retrieval.
    """
 
    def __init__(self) -> None:
        cfg = settings.azure_search
        credential = AzureKeyCredential(cfg.api_key)
 
        self._index_name = cfg.index_name
        self._index_client = SearchIndexClient(
            endpoint=cfg.endpoint,
            credential=credential,
        )
        self._search_client = SearchClient(
            endpoint=cfg.endpoint,
            index_name=self._index_name,
            credential=credential,
        )
 
        logger.info(
            "VectorStore initialised | endpoint=%s | index=%s",
            cfg.endpoint,
            self._index_name,
        )
 
    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def ensure_index_exists(self) -> None:
        """
        Create the Azure AI Search vector index if it does not already exist.
        Safe to call multiple times — no-op if the index exists.
        """
        try:
            self._index_client.get_index(self._index_name)
            logger.info("Index '%s' already exists.", self._index_name)
            return
        except Exception:
            logger.info(
                "Index '%s' not found — creating now.", self._index_name
            )
 
        index = SearchIndex(
            name=self._index_name,
            fields=[
                SimpleField(
                    name=_FIELD_ID,
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name=_FIELD_TEXT,
                    type=SearchFieldDataType.String,
                ),
                SimpleField(
                    name=_FIELD_SOURCE,
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name=_FIELD_INDEX,
                    type=SearchFieldDataType.Int32,
                    filterable=True,
                    sortable=True,
                ),
                SearchField(
                    name=_FIELD_VECTOR,
                    type=SearchFieldDataType.Collection(
                        SearchFieldDataType.Single
                    ),
                    searchable=True,
                    vector_search_dimensions=_VECTOR_DIMENSIONS,
                    vector_search_profile_name=_PROFILE_NAME,
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name=_ALGORITHM_NAME),
                ],
                profiles=[
                    VectorSearchProfile(
                        name=_PROFILE_NAME,
                        algorithm_configuration_name=_ALGORITHM_NAME,
                    ),
                ],
            ),
        )
 
        self._index_client.create_index(index)
        logger.info("Index '%s' created successfully.", self._index_name)
 
    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Upload embedded chunks to Azure AI Search in batches.
 
        Args:
            embedded_chunks: List of EmbeddedChunk objects to store.
            batch_size:      Number of documents per upload batch.
 
        Returns:
            Total number of documents successfully uploaded.
 
        Raises:
            ValueError: If embedded_chunks is empty.
        """
        if not embedded_chunks:
            raise ValueError("No chunks to upload.")
 
        self.ensure_index_exists()
 
        total_uploaded = 0
        batches = [
            embedded_chunks[i : i + batch_size]
            for i in range(0, len(embedded_chunks), batch_size)
        ]
 
        for batch_num, batch in enumerate(batches, start=1):
            documents = [
                {
                    _FIELD_ID: ec.chunk_id,
                    _FIELD_TEXT: ec.text,
                    _FIELD_SOURCE: ec.chunk.source_file,
                    _FIELD_INDEX: ec.chunk.index,
                    _FIELD_VECTOR: ec.embedding,
                }
                for ec in batch
            ]
 
            result = self._search_client.upload_documents(documents)
            succeeded = sum(1 for r in result if r.succeeded)
            total_uploaded += succeeded
 
            logger.info(
                "Upload batch %d/%d | uploaded=%d/%d",
                batch_num,
                len(batches),
                succeeded,
                len(batch),
            )
 
        logger.info(
            "All chunks uploaded | total=%d", total_uploaded
        )
        return total_uploaded
 
    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search against the index.
 
        Args:
            query_vector: Embedding vector of the user query.
            top_k:        Number of top results to return.
            source_file:  Optional filter to restrict search to one document.
 
        Returns:
            List of SearchResult objects ordered by relevance score.
        """
        k = top_k or settings.app.retrieval_top_k
 
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k,
            fields=_FIELD_VECTOR,
        )
 
        filter_expr = None
        if source_file:
            filter_expr = f"{_FIELD_SOURCE} eq '{source_file}'"
 
        logger.info(
            "Vector search | top_k=%d | filter=%s", k, filter_expr or "none"
        )
 
        raw_results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=[_FIELD_ID, _FIELD_TEXT, _FIELD_SOURCE, _FIELD_INDEX],
            top=k,
        )
 
        results = [
            SearchResult(
                chunk_id=r[_FIELD_ID],
                text=r[_FIELD_TEXT],
                source_file=r[_FIELD_SOURCE],
                score=r["@search.score"],
            )
            for r in raw_results
        ]
 
        logger.info("Search returned %d results.", len(results))
        return results
 
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def delete_index(self) -> None:
        """Delete the search index entirely. Use with caution."""
        self._index_client.delete_index(self._index_name)
        logger.warning("Index '%s' deleted.", self._index_name)
 
    def get_document_count(self) -> int:
        """Return the total number of documents in the index."""
        result = self._search_client.get_document_count()
        return result
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
vector_store = VectorStore()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python ingestion/vector_store.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ingestion.chunker import Chunk
    from ingestion.embedder import EmbeddedChunk, Embedder
 
    print("\n=== VectorStore Smoke Test ===\n")
 
    vs = VectorStore()
    emb = Embedder()
 
    print("1. Ensuring index exists...")
    try:
        vs.ensure_index_exists()
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
        sys.exit(1)
 
    print("2. Uploading test chunks...")
    try:
        sample_chunks = [
            Chunk(
                chunk_id="smoke_test_chunk_0000",
                text="Payment is due within 30 days of invoice receipt.",
                index=0,
                char_start=0,
                char_end=50,
                source_file="smoke_test.pdf",
            ),
            Chunk(
                chunk_id="smoke_test_chunk_0001",
                text="Either party may terminate this agreement with 30 days written notice.",
                index=1,
                char_start=50,
                char_end=120,
                source_file="smoke_test.pdf",
            ),
        ]
        embedded = emb.embed_chunks(sample_chunks)
        count = vs.upload_chunks(embedded)
        print(f"   Uploaded: {count} chunks")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("3. Testing vector search...")
    try:
        import time
        time.sleep(2)  # Allow index to refresh
        query_vector = emb.embed_query("What are the payment terms?")
        results = vs.search(query_vector, top_k=2)
        for r in results:
            print(f"   {r}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("4. Document count...")
    try:
        count = vs.get_document_count()
        print(f"   Total documents in index: {count}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")
 