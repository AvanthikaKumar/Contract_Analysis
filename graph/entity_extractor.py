"""
graph/entity_extractor.py
--------------------------
Extracts structured entities and relationships from contract text
using Azure OpenAI and the entity_extraction_prompt.
 
Responsibilities:
- Accept raw contract text or chunks.
- Call the LLM with the entity extraction prompt.
- Parse the JSON response into typed Entity and Relationship objects.
- Return a clean ExtractionResult ready for graph construction.
 
Usage:
    from graph.entity_extractor import entity_extractor
 
    result = entity_extractor.extract(text)
    print(result.entities)
    print(result.relationships)
"""
 
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from core.prompt_manager import prompt_manager
from llm.azure_openai_client import openai_client
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Typed data classes
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    """
    A single entity extracted from the contract.
 
    Attributes:
        id:         Unique snake_case identifier e.g. "acme_corporation"
        type:       Entity type: PARTY | DATE | CLAUSE | FINANCIAL_TERM |
                    OBLIGATION | GOVERNING_LAW
        label:      Human-readable name e.g. "Acme Corporation"
        properties: Additional key-value properties from the contract.
    """
    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
 
    def __repr__(self) -> str:
        return f"Entity(id='{self.id}', type='{self.type}', label='{self.label}')"
 
 
@dataclass
class Relationship:
    """
    A relationship between two entities in the contract graph.
 
    Attributes:
        from_id:    Source entity ID.
        to_id:      Target entity ID.
        type:       Relationship type e.g. "PARTY_TO_CONTRACT"
        properties: Additional key-value properties.
    """
    from_id: str
    to_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
 
    def __repr__(self) -> str:
        return (
            f"Relationship("
            f"'{self.from_id}' --[{self.type}]--> '{self.to_id}')"
        )
 
 
@dataclass
class ExtractionResult:
    """
    Full extraction result from a contract text.
 
    Attributes:
        entities:      List of extracted Entity objects.
        relationships: List of extracted Relationship objects.
        source_file:   Original contract file name.
    """
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    source_file: str = ""
 
    def __repr__(self) -> str:
        return (
            f"ExtractionResult("
            f"entities={len(self.entities)}, "
            f"relationships={len(self.relationships)}, "
            f"source='{self.source_file}')"
        )
 
 
# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------
class EntityExtractor:
    """
    Extracts entities and relationships from contract text using the LLM.
 
    Uses the entity_extraction_prompt.md template to instruct the LLM
    to return structured JSON. Parses and validates the response before
    returning typed ExtractionResult objects.
    """
 
    # Valid entity and relationship types for validation
    _VALID_ENTITY_TYPES = {
        "PARTY", "DATE", "CLAUSE", "FINANCIAL_TERM",
        "OBLIGATION", "GOVERNING_LAW", "PRODUCT", "LOCATION",
    }
    _VALID_RELATIONSHIP_TYPES = {
        # Original
        "PARTY_TO_CONTRACT", "CLAUSE_CONTAINS_OBLIGATION",
        "OBLIGATION_ASSIGNED_TO", "FINANCIAL_TERM_IN_CLAUSE",
        "GOVERNED_BY", "EFFECTIVE_FROM", "EXPIRES_ON",
        # New rich relationships
        "PARTY_IS_BUYER", "PARTY_IS_SELLER",
        "PARTY_HAS_OBLIGATION", "CLAUSE_CONTAINS_FINANCIAL_TERM",
        "PARTY_SUPPLIES", "PARTY_RECEIVES",
        "DELIVERY_LOCATION", "OBLIGATION_TRIGGERS_FINANCIAL_TERM",
        "PARTY_GOVERNED_BY",
    }
 
    def __init__(self) -> None:
        logger.info("EntityExtractor initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        text: str,
        source_file: str = "unknown",
    ) -> ExtractionResult:
        """
        Extract entities and relationships from contract text.
 
        Args:
            text:        Contract text to extract from.
            source_file: Source filename for metadata.
 
        Returns:
            ExtractionResult with entities and relationships.
 
        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot extract entities from empty text.")
 
        logger.info(
            "Starting entity extraction | file=%s | chars=%d",
            source_file,
            len(text),
        )
 
        # Load and render the prompt
        prompt = prompt_manager.load(
            "entity_extraction_prompt",
            variables={"context": text},
        )
 
        # Call LLM
        response = openai_client.get_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a contract entity extraction specialist. "
                        "Always respond with valid JSON only. "
                        "No explanation, no markdown, no code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=6000,
        )
 
        # Parse and validate response
        result = self._parse_response(response, source_file)
 
        logger.info(
            "Extraction complete | file=%s | entities=%d | relationships=%d",
            source_file,
            len(result.entities),
            len(result.relationships),
        )
 
        return result
 
    def extract_from_chunks(
        self,
        chunks: list,
        source_file: str = "unknown",
        max_chunks: int = 20,
    ) -> ExtractionResult:
        """
        Extract entities from multiple chunks, merging and deduplicating results.
 
        For large contracts, we use the first N chunks which typically
        contain the key parties, dates, and clauses.
 
        Args:
            chunks:      List of Chunk objects from the Chunker.
            source_file: Source filename for metadata.
            max_chunks:  Maximum number of chunks to process.
 
        Returns:
            Merged ExtractionResult with deduplicated entities.
        """
        # Use first N chunks — they contain most key contract metadata
        selected = chunks[:max_chunks]
        combined_text = "\n\n".join(c.text for c in selected)
 
        logger.info(
            "Extracting from %d chunks | source=%s", len(selected), source_file
        )
 
        return self.extract(combined_text, source_file)
    
# ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        source_file: str,
    ) -> ExtractionResult:
        """
        Parse LLM JSON response into typed ExtractionResult.
 
        Handles common LLM response issues like markdown fences,
        extra whitespace, and partial JSON.
        """
        # Strip markdown code fences if LLM added them despite instructions
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
 
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM JSON response: %s\nRaw response: %s",
                exc,
                response[:500],
            )
            # Return empty result rather than crashing the pipeline
            return ExtractionResult(source_file=source_file)
 
        entities = self._parse_entities(data.get("entities", []))
 
        # Build a mapping from LLM-generated IDs to our canonical IDs
        # so relationships can reference the correct vertex IDs
        import re as _re
        llm_id_to_canonical = {}
        for item in data.get("entities", []):
            llm_id = str(item.get("id", "")).strip()
            label  = str(item.get("label", "")).strip()
            if llm_id and label:
                canonical = _re.sub(r"[^a-zA-Z0-9_\-]", "_", label).strip("_")[:100]
                if canonical:
                    llm_id_to_canonical[llm_id] = canonical
 
        relationships = self._parse_relationships(
            data.get("relationships", []),
            valid_ids={e.id for e in entities},
            id_map=llm_id_to_canonical,
        )
 
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_file=source_file,
        )
 
    def _parse_entities(self, raw: list[dict]) -> list[Entity]:
        """Parse and validate raw entity dicts into Entity objects."""
        entities = []
        seen_ids = set()
 
        for item in raw:
            try:
                entity_id = str(item.get("id", "")).strip()
                entity_type = str(item.get("type", "")).strip().upper()
                label = str(item.get("label", "")).strip()
 
                if not entity_id or not label:
                    logger.warning("Skipping entity with missing id or label: %s", item)
                    continue
 
                if entity_type not in self._VALID_ENTITY_TYPES:
                    logger.warning(
                        "Unknown entity type '%s' — skipping.", entity_type
                    )
                    continue
 
                if entity_id in seen_ids:
                    logger.debug("Duplicate entity id '%s' — skipping.", entity_id)
                    continue
 
                # Use label as the canonical ID for readable graph display
                # Normalise: lowercase, spaces to underscores, strip special chars
                import re as _re
                canonical_id = _re.sub(r'[^a-zA-Z0-9_\-]', '_', label).strip('_')[:100]
                if not canonical_id:
                    canonical_id = entity_id
 
                if canonical_id in seen_ids:
                    logger.debug("Duplicate canonical id '%s' — skipping.", canonical_id)
                    continue
 
                seen_ids.add(canonical_id)
                entities.append(Entity(
                    id=canonical_id,
                    type=entity_type,
                    label=label,
                    properties=item.get("properties") or {},
                ))
 
            except Exception as exc:
                logger.warning("Failed to parse entity %s: %s", item, exc)
                continue
 
        logger.debug("Parsed %d valid entities.", len(entities))
        return entities
 
    def _parse_relationships(
        self,
        raw: list[dict],
        valid_ids: set[str],
        id_map: dict[str, str] | None = None,
    ) -> list[Relationship]:
        """Parse and validate raw relationship dicts into Relationship objects."""
        relationships = []
        id_map = id_map or {}
 
        for item in raw:
            try:
                # Remap LLM-generated IDs to our canonical label-derived IDs
                raw_from = str(item.get("from_id", "")).strip()
                raw_to   = str(item.get("to_id", "")).strip()
                from_id  = id_map.get(raw_from, raw_from)
                to_id    = id_map.get(raw_to,   raw_to)
                rel_type = str(item.get("type", "")).strip().upper()
 
                if not from_id or not to_id:
                    logger.warning("Skipping relationship with missing ids: %s", item)
                    continue
 
                if rel_type not in self._VALID_RELATIONSHIP_TYPES:
                    logger.warning(
                        "Unknown relationship type '%s' — skipping.", rel_type
                    )
                    continue
 
                # Warn if referencing unknown entities but still include
                if from_id not in valid_ids:
                    logger.warning("Relationship from unknown entity '%s'.", from_id)
                if to_id not in valid_ids:
                    logger.warning("Relationship to unknown entity '%s'.", to_id)
 
                relationships.append(Relationship(
                    from_id=from_id,
                    to_id=to_id,
                    type=rel_type,
                    properties=item.get("properties") or {},
                ))
 
            except Exception as exc:
                logger.warning("Failed to parse relationship %s: %s", item, exc)
                continue
 
        logger.debug("Parsed %d valid relationships.", len(relationships))
        return relationships
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
entity_extractor = EntityExtractor()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python graph/entity_extractor.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== EntityExtractor Smoke Test ===\n")
 
    sample_text = """
    THIS MASTER SERVICES AGREEMENT ("Agreement") is entered into as of January 1, 2025,
    by and between Acme Corporation, a Delaware corporation ("Client"), and TechVendor Inc.,
    a California corporation ("Vendor").
 
    1. SERVICES
    Vendor agrees to provide software development services as described in Exhibit A.
    Services shall commence on February 1, 2025.
 
    2. PAYMENT TERMS
    Client shall pay Vendor a monthly retainer of $50,000, due within 30 days of invoice.
    Late payments shall accrue interest at 1.5% per month.
 
    3. TERMINATION
    Either party may terminate this Agreement with 30 days written notice.
 
    4. GOVERNING LAW
    This Agreement shall be governed by the laws of the State of Delaware.
    """
 
    extractor = EntityExtractor()
 
    print("Extracting entities from sample contract...\n")
    try:
        result = extractor.extract(sample_text.strip(), source_file="test_contract.pdf")
 
        print(f"  Entities found     : {len(result.entities)}")
        print(f"  Relationships found: {len(result.relationships)}")
        print()
 
        print("  --- Entities ---")
        for e in result.entities:
            print(f"  {e}")
 
        print()
        print("  --- Relationships ---")
        for r in result.relationships:
            print(f"  {r}")
 
        print("\n  PASSED\n")
    except Exception as exc:
        print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")