"""
graph/graph_builder.py
"""
 
import logging
import re
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from graph.entity_extractor import Entity, ExtractionResult, Relationship
from graph.graph_client import graph_client
 
logger = logging.getLogger(__name__)
 
 
class GraphBuilder:
 
    def __init__(self) -> None:
        logger.info("GraphBuilder initialised.")
 
    @staticmethod
    def _safe_label(label: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", label).strip("_")[:200]
        return safe if safe else "unknown"
 
    def build(self, extraction_result: ExtractionResult) -> dict[str, int]:
        logger.info(
            "Building graph | source=%s | entities=%d | relationships=%d",
            extraction_result.source_file,
            len(extraction_result.entities),
            len(extraction_result.relationships),
        )
 
        vertices_created = 0
        vertices_skipped = 0
        edges_created    = 0
        edges_skipped    = 0
 
        source_file    = extraction_result.source_file
        contract_label = self._safe_label(Path(source_file).stem)
 
        _TYPE_TO_EDGE = {
            "PARTY":          "HAS_PARTY",
            "DATE":           "HAS_DATE",
            "CLAUSE":         "HAS_CLAUSE",
            "FINANCIAL_TERM": "HAS_FINANCIAL_TERM",
            "OBLIGATION":     "HAS_OBLIGATION",
            "GOVERNING_LAW":  "HAS_GOVERNING_LAW",
        }
 
        # ── Step 1: CONTRACT hub vertex ──────────────────────────────
        ok = self._upsert_vertex(contract_label, "CONTRACT", source_file,
                                 {"file_name": source_file})
        vertices_created += ok; vertices_skipped += (not ok)
 
        # ── Step 2: ENTITY_TYPE shared vertices ──────────────────────
        seen_types: set[str] = set()
        for entity in extraction_result.entities:
            if entity.type not in seen_types:
                seen_types.add(entity.type)
                safe_type = self._safe_label(entity.type)
                ok = self._upsert_vertex(safe_type, "ENTITY_TYPE", "shared",
                                         {"description": entity.type.replace("_", " ").title()})
                vertices_created += ok; vertices_skipped += (not ok)
 
        # ── Step 3: entity value vertices ────────────────────────────
        for entity in extraction_result.entities:
            safe = self._safe_label(entity.label)
            ok = self._upsert_vertex(safe, entity.type, source_file,
                                     entity.properties)
            vertices_created += ok; vertices_skipped += (not ok)
 
        # ── Step 4: edges ─────────────────────────────────────────────
        for entity in extraction_result.entities:
            safe = self._safe_label(entity.label)
            edge_label = _TYPE_TO_EDGE.get(entity.type, "HAS_ENTITY")
 
            # CONTRACT → entity value
            ok = self._upsert_edge(contract_label, edge_label, safe)
            edges_created += ok; edges_skipped += (not ok)
 
            # ENTITY_TYPE → entity value
            safe_type = self._safe_label(entity.type)
            ok = self._upsert_edge(safe_type, "IS_A", safe)
            edges_created += ok; edges_skipped += (not ok)
 
        # ── Step 5: LLM relationships ─────────────────────────────────
        for rel in extraction_result.relationships:
            ok = self._upsert_edge(rel.from_id, rel.type, rel.to_id)
            edges_created += ok; edges_skipped += (not ok)
 
        stats = {
            "vertices_created": vertices_created,
            "vertices_skipped": vertices_skipped,
            "edges_created":    edges_created,
            "edges_skipped":    edges_skipped,
        }
        logger.info("Graph build complete | %s", stats)
        return stats
 
    # ── Vertex upsert — try create, treat 409 as already-exists ──────
    def _upsert_vertex(
        self,
        safe_label: str,
        entity_type: str,
        source_file: str,
        properties: dict,
    ) -> bool:
        """Returns True if created, False if already existed or failed."""
        query = (
            "g.addV(entityType)"
            ".property('id',           safeLabel)"
            ".property('pk',           pk)"
            ".property('entity_label', safeLabel)"
            ".property('entity_type',  entityType)"
            ".property('source_file',  src)"
        )
        bindings: dict = {
            "entityType": entity_type,
            "safeLabel":  safe_label,
            "pk":         source_file,
            "src":        source_file,
        }
        # Attach extra properties
        for i, (k, v) in enumerate(properties.items()):
            if v is not None:
                pk_key = f"pk_{i}"; pv_key = f"pv_{i}"
                query += f".property({pk_key}, {pv_key})"
                bindings[pk_key] = str(k)
                bindings[pv_key] = str(v)
 
        try:
            graph_client.execute(query, bindings=bindings)
            logger.debug("Vertex created: '%s'", safe_label)
            return True
        except Exception as exc:
            msg = str(exc)
            if "409" in msg or "Conflict" in msg or "already exists" in msg.lower():
                logger.debug("Vertex already exists (409): '%s'", safe_label)
                return False
            logger.error("Vertex creation failed for '%s': %s", safe_label, exc)
            return False
 
    # ── Edge upsert — try create, treat 409 as already-exists ────────
    def _upsert_edge(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
    ) -> bool:
        """Returns True if created, False if already existed or failed."""
        try:
            graph_client.execute(
                "g.V(fromId).addE(edgeType).to(g.V(toId))",
                bindings={
                    "fromId":   from_id,
                    "edgeType": edge_type,
                    "toId":     to_id,
                },
            )
            logger.debug("Edge created: %s -[%s]-> %s", from_id, edge_type, to_id)
            return True
        except Exception as exc:
            msg = str(exc)
            if "409" in msg or "Conflict" in msg or "already exists" in msg.lower():
                logger.debug("Edge already exists (409): %s -[%s]-> %s",
                             from_id, edge_type, to_id)
                return False
            # Vertex missing — log warning, not error
            if "not found" in msg.lower() or "404" in msg:
                logger.warning("Edge skipped — vertex missing: %s or %s",
                               from_id, to_id)
                return False
            logger.error("Edge creation failed %s -[%s]-> %s : %s",
                         from_id, edge_type, to_id, exc)
            return False
 
    def get_graph_summary(self) -> dict[str, int]:
        return {
            "total_vertices": graph_client.get_vertex_count(),
            "total_edges":    graph_client.get_edge_count(),
        }
 
    def drop_document_vertices(self, source_file: str) -> int:
        count_result = graph_client.execute(
            "g.V().has('source_file', src).count()",
            bindings={"src": source_file},
        )
        count = int(count_result[0]) if count_result else 0
        graph_client.execute(
            "g.V().has('source_file', src).drop()",
            bindings={"src": source_file},
        )
        return count
 
 
graph_builder = GraphBuilder()
 
 
if __name__ == "__main__":
    from graph.entity_extractor import entity_extractor
 
    print("\n=== GraphBuilder Smoke Test ===\n")
 
    sample_text = """
    THIS MASTER SERVICES AGREEMENT is entered into as of January 1, 2025,
    by and between Acme Corporation ("Client") and TechVendor Inc. ("Vendor").
    Client shall pay Vendor a monthly retainer of $50,000 within 30 days of invoice.
    Late payments shall accrue interest at 1.5% per month.
    Either party may terminate this Agreement with 30 days written notice.
    This Agreement shall be governed by the laws of the State of Delaware.
    """.strip()
 
    builder = GraphBuilder()
 
    print("1. Extracting entities...")
    extraction = entity_extractor.extract(sample_text, source_file="smoke_test.pdf")
    print(f"   Entities: {len(extraction.entities)}")
    for e in extraction.entities:
        print(f"   [{e.type}] {e.label} → {builder._safe_label(e.label)}")
 
    print("\n2. Building graph...")
    stats = builder.build(extraction)
    print(f"   Vertices created : {stats['vertices_created']}")
    print(f"   Vertices skipped : {stats['vertices_skipped']}")
    print(f"   Edges created    : {stats['edges_created']}")
    print(f"   Edges skipped    : {stats['edges_skipped']}")
 
    print("\n3. Graph summary...")
    summary = builder.get_graph_summary()
    print(f"   Total vertices: {summary['total_vertices']}")
    print(f"   Total edges   : {summary['total_edges']}")
 
    print("\n=== Done ===\n")