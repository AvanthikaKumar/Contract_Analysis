"""
graph/graph_client.py
----------------------
Cosmos DB Gremlin API connection client for the Contract Intelligence System.
 
Responsibilities:
- Establish and maintain a connection to Cosmos DB via the Gremlin protocol.
- Expose a single method to execute Gremlin traversal queries.
- Handle connection errors cleanly with retries.
- Provide a safe close() method for cleanup.
 
Usage:
    from graph.graph_client import graph_client
 
    results = graph_client.execute("g.V().limit(5).valueMap()")
"""
 
import logging
import sys
from pathlib import Path
from typing import Any, Optional
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from gremlin_python.driver import client as gremlin_client
from gremlin_python.driver import serializer
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
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
 
 
# ---------------------------------------------------------------------------
# Graph client wrapper
# ---------------------------------------------------------------------------
class GraphClient:
    """
    Manages the Gremlin connection to Cosmos DB and executes graph queries.
 
    Cosmos DB Gremlin API uses WebSocket connections. This client lazily
    initialises the connection on first use and reuses it for all queries.
 
    Attributes:
        _client: The underlying gremlin_python driver client.
        _connected: Whether the connection has been established.
    """
 
    def __init__(self) -> None:
        self._client: Optional[gremlin_client.Client] = None
        self._connected: bool = False
        logger.info("GraphClient created (connection is lazy — opens on first use).")
 
    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """
        Open the Gremlin WebSocket connection to Cosmos DB.
        Safe to call multiple times — no-op if already connected.
        """
        if self._connected and self._client:
            logger.debug("GraphClient already connected.")
            return
 
        cfg = settings.cosmos_db
 
        # Build the traversal source string required by Cosmos DB
        traversal_source = f"g"
 
        logger.info(
            "Connecting to Cosmos DB Gremlin | endpoint=%s | db=%s | graph=%s",
            cfg.gremlin_endpoint,
            cfg.database_name,
            cfg.graph_name,
        )
 
        self._client = gremlin_client.Client(
            url=cfg.gremlin_endpoint,
            traversal_source=traversal_source,
            username=f"/dbs/{cfg.database_name}/colls/{cfg.graph_name}",
            password=cfg.primary_key,
            message_serializer=serializer.GraphSONSerializersV2d0(),
        )
 
        self._connected = True
        logger.info("Cosmos DB Gremlin connection established.")
 
    def close(self) -> None:
        """Close the Gremlin connection cleanly."""
        if self._client and self._connected:
            self._client.close()
            self._connected = False
            self._client = None
            logger.info("Cosmos DB Gremlin connection closed.")
 
    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    @retry(**_RETRY_POLICY)
    def execute(
        self,
        query: str,
        bindings: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """
        Execute a Gremlin traversal query and return all results.
 
        Args:
            query:    Gremlin query string e.g. "g.V().limit(10).valueMap()"
            bindings: Optional parameter bindings for the query.
 
        Returns:
            List of result objects from the traversal.
 
        Raises:
            Exception: If the query fails after retries.
        """
        if not self._connected:
            self.connect()
 
        logger.debug("Executing Gremlin query: %s", query)
 
        callback = self._client.submitAsync(query, bindings=bindings or {})
        results = callback.result().all().result()
 
        logger.debug("Gremlin query returned %d results.", len(results))
        return results
 
    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def vertex_exists(self, vertex_id: str) -> bool:
        """
        Check whether a vertex with the given ID exists in the graph.
 
        Args:
            vertex_id: The unique vertex ID to check.
 
        Returns:
            True if the vertex exists, False otherwise.
        """
        results = self.execute(
            "g.V().has('id', vertexId).count()",
            bindings={"vertexId": vertex_id},
        )
        return bool(results and results[0] > 0)
 
    def drop_graph(self) -> None:
        """
        Drop ALL vertices and edges from the graph.
        Use with caution — intended for testing and resets only.
        """
        logger.warning("Dropping all vertices and edges from the graph.")
        self.execute("g.V().drop()")
        logger.info("Graph cleared.")
 
    def get_vertex_count(self) -> int:
        """Return total number of vertices in the graph."""
        results = self.execute("g.V().count()")
        return int(results[0]) if results else 0
 
    def get_edge_count(self) -> int:
        """Return total number of edges in the graph."""
        results = self.execute("g.E().count()")
        return int(results[0]) if results else 0
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
graph_client = GraphClient()
 
 
# ---------------------------------------------------------------------------
# Smoke test — run directly: python graph/graph_client.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== GraphClient Smoke Test ===\n")
 
    client = GraphClient()
 
    print("1. Testing connection to Cosmos DB Gremlin...")
    try:
        client.connect()
        print("   Connection established. PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
        sys.exit(1)
 
    print("2. Testing vertex count query...")
    try:
        count = client.get_vertex_count()
        print(f"   Vertex count: {count}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("3. Testing edge count query...")
    try:
        count = client.get_edge_count()
        print(f"   Edge count: {count}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("4. Testing raw Gremlin query...")
    try:
        results = client.execute("g.V().limit(3).valueMap()")
        print(f"   Sample vertices returned: {len(results)}")
        for v in results:
            print(f"   {v}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    client.close()
    print("=== Smoke test complete ===\n")