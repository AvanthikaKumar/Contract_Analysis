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
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(label)).strip("_")[:200]
        return safe if safe else "unknown"
 
    def _vertex_exists(self, vid: str) -> bool:
        try:
            result = graph_client.execute(
                "g.V().hasId(vid).count()",
                bindings={"vid": vid},
            )
            return bool(result and result[0] > 0)
        except Exception as e:
            logger.warning("Vertex existence check failed for '%s': %s", vid, e)
            return False
 
    def _edge_exists(self, from_id: str, rel_type: str, to_id: str) -> bool:
        try:
            result = graph_client.execute(
                "g.V().hasId(fid).out(etype).hasId(tid).count()",
                bindings={"fid": from_id, "etype": rel_type, "tid": to_id},
            )
            return bool(result and result[0] > 0)
        except Exception as e:
            logger.warning("Edge existence check failed: %s", e)
            return False
 
    def build(self, extraction_result: ExtractionResult) -> dict[str, int]:
 
        vertices_created = 0
        vertices_skipped = 0
        edges_created    = 0
        edges_skipped    = 0
 
        source_file     = extraction_result.source_file
        contract_label  = self._safe_label(Path(source_file).stem)
 
        _TYPE_TO_EDGE = {
            "PARTY":          "HAS_PARTY",
            "DATE":           "HAS_DATE",
            "CLAUSE":         "HAS_CLAUSE",
            "FINANCIAL_TERM": "HAS_FINANCIAL_TERM",
            "OBLIGATION":     "HAS_OBLIGATION",
            "GOVERNING_LAW":  "HAS_GOVERNING_LAW",
        }
 
        # ── Step 1: CONTRACT hub vertex ───────────────────────────────
        if self._vertex_exists(contract_label):
            vertices_skipped += 1
        else:
            try:
                graph_client.execute(
                    "g.addV('CONTRACT')"
                    ".property('id', vid)"
                    ".property('pk', vpk)"
                    ".property('entity_label', vid)"
                    ".property('entity_type', 'CONTRACT')"
                    ".property('source_file', src)",
                    bindings={
                        "vid": contract_label,
                        "vpk": contract_label,
                        "src": source_file,
                    },
                )
                vertices_created += 1
                logger.debug("Created CONTRACT vertex: %s", contract_label)
            except Exception as e:
                logger.error("Failed to create CONTRACT vertex: %s", e)
 
        # ── Step 2: Entity value vertices ─────────────────────────────
        for entity in extraction_result.entities:
            vid = self._safe_label(entity.label)
            if self._vertex_exists(vid):
                vertices_skipped += 1
                continue
            try:
                graph_client.execute(
                    "g.addV(etype)"
                    ".property('id', vid)"
                    ".property('pk', vpk)"
                    ".property('entity_label', vid)"
                    ".property('entity_type', etype)"
                    ".property('source_file', src)",
                    bindings={
                        "etype": entity.type,
                        "vid":   vid,
                        "vpk":   contract_label,
                        "src":   source_file,
                    },
                )
                vertices_created += 1
                logger.debug("Created vertex: %s [%s]", vid, entity.type)
            except Exception as e:
                logger.error("Failed to create vertex '%s': %s", vid, e)
 
        # ── Step 3: CONTRACT → entity edges ───────────────────────────
        for entity in extraction_result.entities:
            vid        = self._safe_label(entity.label)
            edge_label = _TYPE_TO_EDGE.get(entity.type, "HAS_ENTITY")
 
            if self._edge_exists(contract_label, edge_label, vid):
                edges_skipped += 1
            else:
                try:
                    graph_client.execute(
                        "g.V().hasId(fid).addE(etype).to(g.V().hasId(tid))",
                        bindings={
                            "fid":   contract_label,
                            "etype": edge_label,
                            "tid":   vid,
                        },
                    )
                    edges_created += 1
                    logger.debug("Edge: %s -[%s]-> %s", contract_label, edge_label, vid)
                except Exception as e:
                    logger.error("Failed to create edge %s->%s: %s", contract_label, vid, e)
 
        stats = {
            "vertices_created": vertices_created,
            "vertices_skipped": vertices_skipped,
            "edges_created":    edges_created,
            "edges_skipped":    edges_skipped,
        }
        logger.info("Graph build complete | %s", stats)
        return stats
 
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
    THIS AGREEMENT is entered into as of January 1, 2025,
    by and between Acme Corporation ("Client") and TechVendor Inc. ("Vendor").
    Payment of $50,000 monthly within 30 days. Governed by laws of Delaware.
    """.strip()
 
    extraction = entity_extractor.extract(sample_text, source_file="smoke_test.pdf")
    print(f"Entities: {len(extraction.entities)}")
    for e in extraction.entities:
        print(f"  [{e.type}] {e.label} → id='{GraphBuilder._safe_label(e.label)}'")
 
    stats = graph_builder.build(extraction)
    print(f"\nVertices created : {stats['vertices_created']}")
    print(f"Vertices skipped : {stats['vertices_skipped']}")
    print(f"Edges created    : {stats['edges_created']}")
    print(f"Edges skipped    : {stats['edges_skipped']}")
 
    summary = graph_builder.get_graph_summary()
    print(f"\nTotal vertices: {summary['total_vertices']}")
    print(f"Total edges   : {summary['total_edges']}")
    print("\n=== Done ===\n")