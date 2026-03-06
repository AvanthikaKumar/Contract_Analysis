 
"""
llm/answer_generator.py
------------------------
Generates context-grounded answers to user queries using Azure OpenAI.
 
Responsibilities:
- Load the appropriate prompt from PromptManager.
- Detect out-of-scope queries using the scope guard prompt.
- Generate answers grounded strictly in the retrieved contract context.
- Return "Not specified in the provided document." when context is insufficient.
- Support contract summarisation as a special query type.
 
Usage:
    from llm.answer_generator import answer_generator
 
    answer = answer_generator.generate(
        query="What are the payment terms?",
        context="[Excerpt 1]... payment is due within 30 days..."
    )
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
 
from core.prompt_manager import prompt_manager
from llm.azure_openai_client import openai_client
 
logger = logging.getLogger(__name__)
 
# Exact phrase returned when the answer is not in the context
OUT_OF_CONTEXT_RESPONSE = "Not specified in the provided document."
OUT_OF_SCOPE_RESPONSE = (
    "This question is outside the scope of contract analysis. "
    "Please ask a question related to the uploaded contract."
)
 
 
# ---------------------------------------------------------------------------
# Answer result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnswerResult:
    """
    The result of an answer generation operation.
 
    Attributes:
        query:        The original user query.
        answer:       The generated answer text.
        context_used: The contract context used to generate the answer.
        is_out_of_scope: Whether the query was flagged as out of scope.
        is_grounded:  Whether the answer was grounded in the context.
    """
    query: str
    answer: str
    context_used: str
    is_out_of_scope: bool = False
    is_grounded: bool = True
 
    def __repr__(self) -> str:
        return (
            f"AnswerResult("
            f"query='{self.query[:50]}', "
            f"out_of_scope={self.is_out_of_scope}, "
            f"answer_chars={len(self.answer)})"
        )
 
 
# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------
class AnswerGenerator:
    """
    Generates grounded answers to contract questions using Azure OpenAI.
 
    Uses governed prompts loaded via PromptManager. Applies a scope guard
    to filter irrelevant queries before calling the main LLM.
    """
 
    def __init__(self) -> None:
        logger.info("AnswerGenerator initialised.")
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        context: str,
        skip_scope_check: bool = False,
    ) -> AnswerResult:
        """
        Generate a grounded answer for a user query.
 
        Args:
            query:            The user's natural language question.
            context:          Retrieved contract chunks as a combined string.
            skip_scope_check: Skip the scope guard (e.g. for summarisation).
 
        Returns:
            AnswerResult with the generated answer and metadata.
 
        Raises:
            ValueError: If query or context is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
 
        if not context or not context.strip():
            logger.warning("No context provided — returning out-of-context response.")
            return AnswerResult(
                query=query,
                answer=OUT_OF_CONTEXT_RESPONSE,
                context_used="",
                is_grounded=False,
            )
 
        # Step 1 — Scope guard check
        if not skip_scope_check:
            is_in_scope = self._check_scope(query)
            if not is_in_scope:
                logger.info("Query flagged as out of scope: '%s'", query[:80])
                return AnswerResult(
                    query=query,
                    answer=OUT_OF_SCOPE_RESPONSE,
                    context_used="",
                    is_out_of_scope=True,
                    is_grounded=False,
                )
 
        # Step 2 — Generate grounded answer
        answer = self._generate_answer(query, context)
 
        # Step 3 — Determine if answer is grounded or not found
        is_grounded = OUT_OF_CONTEXT_RESPONSE.lower() not in answer.lower()
 
        logger.info(
            "Answer generated | grounded=%s | chars=%d",
            is_grounded,
            len(answer),
        )
 
        return AnswerResult(
            query=query,
            answer=answer,
            context_used=context,
            is_grounded=is_grounded,
        )
 
    def summarise(self, context: str) -> AnswerResult:
        """
        Generate a structured executive summary of the contract.
 
        Args:
            context: Full contract text or combined chunk text.
 
        Returns:
            AnswerResult with the structured summary.
        """
        if not context or not context.strip():
            raise ValueError("Context cannot be empty for summarisation.")
 
        logger.info("Generating contract summary | chars=%d", len(context))
 
        prompt = prompt_manager.load(
            "summarization_prompt",
            variables={"context": context},
        )
 
        summary = openai_client.get_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior legal analyst. "
                        "Produce structured contract summaries in clear business language."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
 
        logger.info("Summary generated | chars=%d", len(summary))
 
        return AnswerResult(
            query="Summarise the contract.",
            answer=summary,
            context_used=context,
            is_grounded=True,
        )
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_scope(self, query: str) -> bool:
        """
        Use the scope guard prompt to classify a query as IN or OUT of scope.
 
        Returns True if the query is in scope, False otherwise.
        """
        try:
            prompt = prompt_manager.load(
                "scope_guard_prompt",
                variables={"question": query},
            )
 
            response = openai_client.get_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query classifier. "
                            "Respond with only IN_SCOPE or OUT_OF_SCOPE."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
 
            classification = response.strip().upper()
            logger.debug("Scope classification: '%s'", classification)
            return "IN_SCOPE" in classification
 
        except Exception as exc:
            # On any error, default to allowing the query through
            logger.warning("Scope check failed — defaulting to IN_SCOPE: %s", exc)
            return True
 
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate a grounded answer using the answer prompt.
 
        Args:
            query:   The user question.
            context: Retrieved contract context.
 
        Returns:
            Generated answer string.
        """
        prompt = prompt_manager.load(
            "answer_prompt",
            variables={
                "context": context,
                "question": query,
            },
        )
 
        answer = openai_client.get_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise contract analysis assistant. "
                        "Answer questions strictly based on the provided context. "
                        f"If the answer is not in the context, respond exactly with: "
                        f"'{OUT_OF_CONTEXT_RESPONSE}'"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
 
        return answer.strip()
 
 
# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
answer_generator = AnswerGenerator()
 
 
# ---------------------------------------------------------------------------
# Smoke test — python llm/answer_generator.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== AnswerGenerator Smoke Test ===\n")
 
    sample_context = """
[Excerpt 1 — Source: test_contract.pdf]
THIS MASTER SERVICES AGREEMENT is entered into as of January 1, 2025,
by and between Acme Corporation ("Client") and TechVendor Inc. ("Vendor").
 
---
 
[Excerpt 2 — Source: test_contract.pdf]
PAYMENT TERMS: Client shall pay Vendor a monthly retainer of $50,000,
due within 30 days of invoice. Late payments shall accrue interest at
1.5% per month.
 
---
 
[Excerpt 3 — Source: test_contract.pdf]
TERMINATION: Either party may terminate this Agreement with 30 days
written notice. Upon termination, Client shall pay all outstanding invoices
within 15 days.
    """.strip()
 
    gen = AnswerGenerator()
 
    test_cases = [
        ("Who are the parties in this agreement?", False),
        ("What are the payment terms?", False),
        ("What is the termination notice period?", False),
        ("What is the weather in London today?", False),
        ("What is the capital of France?", False),
    ]
 
    for query, skip_scope in test_cases:
        print(f"Query: '{query}'")
        try:
            result = gen.generate(query, sample_context, skip_scope_check=skip_scope)
            print(f"  Out of scope : {result.is_out_of_scope}")
            print(f"  Grounded     : {result.is_grounded}")
            print(f"  Answer       : {result.answer[:200]}")
            print(f"  PASSED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
 
    print("--- Testing summarise() ---\n")
  
    try:
        summary = gen.summarise(sample_context)
        print(f"  Summary chars: {len(summary.answer)}")
        print(f"  Preview      : {summary.answer[:300]}")
        print(f"  PASSED\n")
    except Exception as exc:
        print(f"  FAILED: {exc}\n")
 
    print("=== Smoke test complete ===\n")