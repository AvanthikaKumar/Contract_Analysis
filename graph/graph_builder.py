"""
graph/graph_builder.py
-----------------------
Builds the contract knowledge graph in Cosmos DB (Gremlin API)
from extracted entities and relationships.
 
Responsibilities:
- Create vertex (node) for each extracted entity.
- Create edge for each extracted relationship.
- Skip duplicates — safe to re-run on the same document.
- Attach all entity properties to the corresponding vertex.
 
Usage:
    from graph.graph_builder import graph_builder
 
    graph_builder.build(extraction_result)
"""
 
import logging
import sys
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from graph.entity_extractor import Entity, ExtractionResult, Relationship
from graph.graph_client import graph_client
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
class GraphBuilder:
    """
    Translates ExtractionResult objects into Cosmos DB Gremlin vertices
    and edges.
 
    Each entity becomes a vertex with its properties attached.
    Each relationship becomes a directed edge between two vertices.
    Existing vertices and edges are skipped to allow safe re-runs.
    """
 
    def __init__(self) -> None:
        logger.info("GraphBuilder initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, extraction_result: ExtractionResult) -> dict[str, int]:
        """
        Build a hub-and-spoke graph from an ExtractionResult.
 
        The contract document is the central hub vertex.
        Every extracted entity is a spoke connected to the hub
        via a BELONGS_TO edge. This makes the graph readable and
        consistent across multiple uploaded documents.
 
        Args:
            extraction_result: Result from EntityExtractor.extract().
 
        Returns:
            Dict with counts: {"vertices_created", "edges_created",
                               "vertices_skipped", "edges_skipped"}
        """
        logger.info(
            "Building graph | source=%s | entities=%d | relationships=%d",
            extraction_result.source_file,
            len(extraction_result.entities),
            len(extraction_result.relationships),
        )
 
        vertices_created = 0
        vertices_skipped = 0
        edges_created = 0
        edges_skipped = 0
 
        source_file = extraction_result.source_file
 
        # Step 1 — Create the CONTRACT hub vertex for this document
        from pathlib import Path as _Path
        contract_label = _Path(source_file).stem
        contract_entity = Entity(
            id=contract_label,
            type="CONTRACT",
            label=contract_label,
            properties={"file_name": source_file},
        )
        created = self._upsert_vertex(contract_entity, source_file)
        if created:
            vertices_created += 1
        else:
            vertices_skipped += 1
 
        # Entity type → edge label mapping
        # These are the FIXED spoke types shown in every contract graph
        _TYPE_TO_EDGE = {
            "PARTY":          "HAS_PARTY",
            "DATE":           "HAS_DATE",
            "CLAUSE":         "HAS_CLAUSE",
            "FINANCIAL_TERM": "HAS_FINANCIAL_TERM",
            "OBLIGATION":     "HAS_OBLIGATION",
            "GOVERNING_LAW":  "HAS_GOVERNING_LAW",
        }
 
        
        import re as _re
 
        # Step 2 — Create one FIXED TYPE vertex per entity type (shared across docs)
        # e.g. a single "SELLER" node, a single "START_DATE" node etc.
        seen_types = set()
        for entity in extraction_result.entities:
            if entity.type not in seen_types:
                seen_types.add(entity.type)
                type_vertex = Entity(
                    id=entity.type,
                    type="ENTITY_TYPE",
                    label=entity.type,
                    properties={"description": entity.type.replace("_", " ").title()},
                )
                created = self._upsert_vertex(type_vertex, "shared")
                if created:
                    vertices_created += 1
                else:
                    vertices_skipped += 1
 
        # Step 3 — Create value vertices for each extracted entity
        for entity in extraction_result.entities:
            created = self._upsert_vertex(entity, source_file)
            if created:
                vertices_created += 1
            else:
                vertices_skipped += 1
 
        # Step 4 — Connect: CONTRACT ──edge──> TYPE ──edge──> VALUE
        for entity in extraction_result.entities:
            edge_label = _TYPE_TO_EDGE.get(entity.type, "HAS_ENTITY")
 
            # Edge 1: CONTRACT → VALUE (direct connection)
            contract_to_value = Relationship(
                from_id=contract_label,
                to_id=entity.label,
                type=edge_label,
            )
            created = self._upsert_edge(contract_to_value)
            if created:
                edges_created += 1
            else:
                edges_skipped += 1
 
            # Edge 2: TYPE_NODE → VALUE (groups same types across docs)
            type_to_value = Relationship(
                from_id=entity.type,
                to_id=entity.label,
                type="IS_A",
            )
            created = self._upsert_edge(type_to_value)
            if created:
                edges_created += 1
            else:
                edges_skipped += 1
 
        # Step 5 — LLM-extracted entity-to-entity relationships
        for relationship in extraction_result.relationships:
            created = self._upsert_edge(relationship)
            if created:
                edges_created += 1
            else:
                edges_skipped += 1
 
        stats = {
            "vertices_created": vertices_created,
            "vertices_skipped": vertices_skipped,
            "edges_created": edges_created,
            "edges_skipped": edges_skipped,
        }
 
        logger.info("Graph build complete | %s", stats)
        return stats
 
    # ------------------------------------------------------------------
    # Vertex management
    # ------------------------------------------------------------------
    def _upsert_vertex(
        self,
        entity: Entity,
        source_file: str,
    ) -> bool:
        """
        Create a vertex if it does not already exist.
 
        Returns True if created, False if already existed.
        """
        # Check if vertex already exists
        import re as _re
        safe_label_check = _re.sub(r"[^a-zA-Z0-9_\-]", "_", entity.label).strip("_")[:200]
 
        existing = graph_client.execute(
            "g.V().has('entity_label', elabel).count()",
            bindings={"elabel": entity.label},
        )
 
        if existing and existing[0] > 0:
            logger.debug("Vertex already exists: '%s'", entity.label)
            return False
 
        # Build vertex — 'id' must be quoted string in Cosmos DB Gremlin
        # Sanitise label for use as vertex id (no special chars)
        import re as _re
        safe_label = _re.sub(r"[^a-zA-Z0-9_\-]", "_", entity.label).strip("_")[:200]
        if not safe_label:
            safe_label = entity.id
 
        query = (
            "g.addV(entityType)"
            ".property('id', safeLabel)"
            ".property('pk', pk)"
            ".property('entity_id', eid)"
            ".property('entity_label', elabel)"
            ".property('entity_type', entityType)"
            ".property('source_file', src)"
        )
 
        bindings: dict = {
            "entityType": entity.type,
            "pk": source_file,
            "eid": entity.id,
            "elabel": entity.label,
            "safeLabel": safe_label,
            "src": source_file,
        }
 
        # Add extra properties dynamically
        for i, (key, value) in enumerate(entity.properties.items()):
            if value is not None:
                prop_key = f"prop_key_{i}"
                prop_val = f"prop_val_{i}"
                query += f".property({prop_key}, {prop_val})"
                bindings[prop_key] = str(key)
                bindings[prop_val] = str(value)
 
        graph_client.execute(query, bindings=bindings)
        logger.debug("Vertex created: '%s' [%s]", entity.label, entity.type)
        return True
    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------
    def _upsert_edge(self, relationship: Relationship) -> bool:
        """
        Create a directed edge between two vertices if it does not exist.
 
        Returns True if created, False if already existed.
        """
        # Check if edge already exists between these two vertices
        existing = graph_client.execute(
            "g.V().has('entity_label', fromId).outE(relType).where(__.inV().has('entity_label', toId)).count()",
            bindings={
                "fromId": relationship.from_id,
                "relType": relationship.type,
                "toId": relationship.to_id,
            },
        )
 
        if existing and existing[0] > 0:
            logger.debug(
                "Edge already exists: '%s' -[%s]-> '%s'",
                relationship.from_id,
                relationship.type,
                relationship.to_id,
            )
            return False
 
        # Verify both vertices exist before creating the edge
        from_exists = graph_client.execute(
            "g.V().has('entity_label', eid).count()",
            bindings={"eid": relationship.from_id},
        )
        to_exists = graph_client.execute(
            "g.V().has('entity_label', eid).count()",
            bindings={"eid": relationship.to_id},
        )
 
        if not (from_exists and from_exists[0] > 0):
            logger.warning(
                "Cannot create edge — source vertex missing: '%s'",
                relationship.from_id,
            )
            return False
 
        if not (to_exists and to_exists[0] > 0):
            logger.warning(
                "Cannot create edge — target vertex missing: '%s'",
                relationship.to_id,
            )
            return False
 
        # Create the directed edge
        graph_client.execute(
            "g.V().has('entity_label', fromId).addE(relType).to(__.V().has('entity_label', toId))",
            bindings={
                "fromId": relationship.from_id,
                "relType": relationship.type,
                "toId": relationship.to_id,
            },
        )
 
        logger.debug(
            "Edge created: '%s' -[%s]-> '%s'",
            relationship.from_id,
            relationship.type,
            relationship.to_id,
        )
        return True
 
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_graph_summary(self) -> dict[str, int]:
        """Return total vertex and edge counts from the graph."""
        return {
            "total_vertices": graph_client.get_vertex_count(),
            "total_edges": graph_client.get_edge_count(),
        }
 
    def drop_document_vertices(self, source_file: str) -> int:
        """
        Remove all vertices belonging to a specific source document.
        Useful for re-ingesting an updated contract.
 
        Args:
            source_file: The source filename to remove vertices for.
 
        Returns:
            Number of vertices dropped.
        """
        count_result = graph_client.execute(
            "g.V().has('source_file', src).count()",
            bindings={"src": source_file},
        )
        count = int(count_result[0]) if count_result else 0
 
        graph_client.execute(
            "g.V().has('source_file', src).drop()",
            bindings={"src": source_file},
        )
 
        logger.info(
            "Dropped %d vertices for source: '%s'", count, source_file
        )
        return count
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
graph_builder = GraphBuilder()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python graph/graph_builder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from graph.entity_extractor import entity_extractor
 
    print("\n=== GraphBuilder Smoke Test ===\n")
 
    sample_text = """
    THIS MASTER SERVICES AGREEMENT is entered into as of January 1, 2025,
    by and between Acme Corporation ("Client") and TechVendor Inc. ("Vendor").
 
    1. PAYMENT TERMS
    Client shall pay Vendor a monthly retainer of $50,000 within 30 days of invoice.
    Late payments shall accrue interest at 1.5% per month.
 
    2. TERMINATION
    Either party may terminate this Agreement with 30 days written notice.
 
    3. GOVERNING LAW
    This Agreement shall be governed by the laws of the State of Delaware.
    """.strip()
 
    builder = GraphBuilder()
 
    print("1. Extracting entities from sample text...")
    try:
        extraction = entity_extractor.extract(
            text=sample_text,
            source_file="smoke_test_contract.pdf",
        )
        print(f"   Entities found     : {len(extraction.entities)}")
        print(f"   Relationships found: {len(extraction.relationships)}")
        for e in extraction.entities:
            print(f"   [{e.type}] {e.label}")
        print()
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
        sys.exit(1)
 
    print("2. Building graph...")
    try:
        stats = builder.build(extraction)
        print(f"   Vertices created : {stats['vertices_created']}")
        print(f"   Vertices skipped : {stats['vertices_skipped']}")
        print(f"   Edges created    : {stats['edges_created']}")
        print(f"   Edges skipped    : {stats['edges_skipped']}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("3. Graph summary...")
    try:
        summary = builder.get_graph_summary()
        print(f"   Total vertices: {summary['total_vertices']}")
        print(f"   Total edges   : {summary['total_edges']}")
        print("   PASSED\n")
    except Exception as exc:
        print(f"   FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")