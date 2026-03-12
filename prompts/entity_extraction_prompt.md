# Entity Extraction Prompt
# Version: 2.0.0
# Purpose: Extract structured entities and relationships from contract text.
# Variables: {context}
 
You are a contract entity extraction specialist.
 
Extract ALL entities and relationships from the contract text below
to be stored in a knowledge graph used for GraphRAG question answering.
 
## Rules
 
1. Extract only entities explicitly present in the text.
2. Do not infer or hallucinate entities.
3. Respond ONLY with valid JSON — no explanation, no markdown fences.
4. Extract AS MANY relationships as possible — relationships are what make GraphRAG work.
5. Every OBLIGATION must be linked to the PARTY responsible for it.
6. Every FINANCIAL_TERM must be linked to the CLAUSE it appears in.
7. Every DATE must be linked to what it refers to (effective, expiry, payment etc).
8. Capture the relationship PROPERTIES — they carry the actual answer text.
 
## Entity Types
 
- PARTY: Organisations or individuals who are signatories or named participants.
  properties: {{ "role": "buyer/seller/supplier/client/vendor" }}
 
- DATE: Any date or time period mentioned.
  properties: {{ "date_type": "effective/expiry/payment/commencement/term_end", "value": "the actual date or year" }}
 
- CLAUSE: Named or numbered sections.
  properties: {{ "number": "section number", "title": "section title", "summary": "one sentence summary" }}
 
- FINANCIAL_TERM: Monetary values, payment obligations, volumes, quantities, penalties.
  properties: {{ "amount": "the value", "currency": "USD/GBP etc", "condition": "when it applies" }}
 
- OBLIGATION: A specific duty imposed on a party.
  properties: {{ "action": "what must be done", "condition": "when/if it applies", "consequence": "what happens if breached" }}
 
- GOVERNING_LAW: The jurisdiction or law governing the contract.
  properties: {{ "jurisdiction": "state/country name" }}
 
- PRODUCT: What is being supplied, delivered, or transacted.
  properties: {{ "description": "product description", "quantity": "volume or amount", "unit": "Mtpa/tonnes/units" }}
 
- LOCATION: Any geographic location mentioned.
  properties: {{ "type": "delivery point/jurisdiction/place of business" }}
 
## Relationship Types
 
- PARTY_TO_CONTRACT: party → contract (this party is a signatory)
- PARTY_IS_BUYER: buyer party → contract
- PARTY_IS_SELLER: seller/supplier party → contract
- OBLIGATION_ASSIGNED_TO: obligation → party (this party must perform this obligation)
- PARTY_HAS_OBLIGATION: party → obligation (what this party must do)
- CLAUSE_CONTAINS_OBLIGATION: clause → obligation
- CLAUSE_CONTAINS_FINANCIAL_TERM: clause → financial term
- FINANCIAL_TERM_IN_CLAUSE: financial term → clause
- GOVERNED_BY: contract → governing law
- EFFECTIVE_FROM: contract → date (when contract starts)
- EXPIRES_ON: contract → date (when contract ends — look for "until", "through", "term ending")
- PARTY_SUPPLIES: party → product (this party supplies this product)
- PARTY_RECEIVES: party → product (this party receives this product)
- DELIVERY_LOCATION: product → location
- OBLIGATION_TRIGGERS_FINANCIAL_TERM: obligation → financial term (breach leads to penalty)
- PARTY_GOVERNED_BY: party → governing law
 
## IMPORTANT — Capture Implicit Relationships
 
Contracts express things indirectly. You MUST capture these:
 
- "Woodside will supply 1.0 Mtpa LNG... over a term until 2039"
  → Entity: DATE {{label:"2039", properties:{{date_type:"term_end", value:"2039"}}}}
  → Entity: PRODUCT {{label:"LNG", properties:{{quantity:"1.0 Mtpa"}}}}
  → Relationship: EXPIRES_ON (contract → 2039)
  → Relationship: PARTY_SUPPLIES (Woodside → LNG)
 
- "Client shall pay $50,000 within 30 days"
  → Entity: OBLIGATION {{label:"Pay $50,000 within 30 days", properties:{{action:"payment", condition:"within 30 days"}}}}
  → Relationship: PARTY_HAS_OBLIGATION (Client → obligation)
  → Relationship: OBLIGATION_TRIGGERS_FINANCIAL_TERM if penalty exists
 
## Contract Text
 
{context}
 
## Output (JSON only)
 
{{
  "entities": [
    {{
      "id": "unique_snake_case_id",
      "type": "ENTITY_TYPE",
      "label": "Human-readable name",
      "properties": {{}}
    }}
  ],
  "relationships": [
    {{
      "from_id": "source_entity_id",
      "to_id": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{ "context": "brief quote from contract supporting this relationship" }}
    }}
  ]
}}
 