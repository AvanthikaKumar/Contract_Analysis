"""
llm/answer_generator.py
------------------------
LangChain-powered answer generator for the Contract Intelligence System.
 
Replaces the custom Azure OpenAI wrapper with LangChain ChatOpenAI chains.
 
Uses:
- ChatOpenAI (LangChain) for chat completions
- PromptTemplate (LangChain) for prompt composition
- LangChain chain: prompt | llm | StrOutputParser
"""
 
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
from config.settings import settings
from core.prompt_manager import prompt_manager
 
logger = logging.getLogger(__name__)
 
OUT_OF_CONTEXT_RESPONSE = "Not specified in the provided document."
OUT_OF_SCOPE_RESPONSE = (
    "This question is outside the scope of contract analysis. "
    "Please ask a question related to the uploaded contract."
)
 
 
@dataclass
class AnswerResult:
    query:           str
    answer:          str
    context_used:    str
    is_out_of_scope: bool = False
    is_grounded:     bool = True
 
 
class AnswerGenerator:
    """
    LangChain-based answer generator.
 
    Builds three chains using LangChain's pipe syntax (prompt | llm | parser):
      1. scope_chain   — IN_SCOPE / OUT_OF_SCOPE classifier
      2. answer_chain  — grounded contract Q&A
      3. summary_chain — executive contract summary
    """
 
    def __init__(self) -> None:
        cfg = settings.azure_openai
 
        # ── LangChain AzureChatOpenAI LLM ──────────────────────────────
        self._llm = AzureChatOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_deployment=cfg.chat_deployment,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            temperature=0.0,
            max_tokens=1500,
        )
 
        self._llm_short = AzureChatOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_deployment=cfg.chat_deployment,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            temperature=0.0,
            max_tokens=10,
        )
 
        parser = StrOutputParser()
 
        # ── Chain 1: Scope guard ────────────────────────────────────────
        scope_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query classifier. Respond with only IN_SCOPE or OUT_OF_SCOPE."),
            ("human",  "{scope_prompt}"),
        ])
        self._scope_chain = scope_prompt | self._llm_short | parser
 
        # ── Chain 2: Answer ─────────────────────────────────────────────
        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a precise contract analysis assistant. "
                "Answer questions strictly based on the provided context. "
                f"If the answer is not in the context, respond exactly with: '{OUT_OF_CONTEXT_RESPONSE}'",
            ),
            ("human", "{answer_prompt}"),
        ])
        self._answer_chain = answer_prompt | self._llm | parser
 
        # ── Chain 3: Summary ────────────────────────────────────────────
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior legal analyst. "
                "Produce structured contract summaries in clear business language.",
            ),
            ("human", "{summary_prompt}"),
        ])
        self._summary_chain = summary_prompt | self._llm | parser
 
        logger.info("AnswerGenerator initialised (LangChain chains).")
 
    # ── Public API ─────────────────────────────────────────────────────
    def generate(
        self,
        query: str,
        context: str,
        skip_scope_check: bool = False,
    ) -> AnswerResult:
 
        if not query.strip():
            raise ValueError("Query cannot be empty.")
 
        if not context.strip():
            return AnswerResult(
                query=query, answer=OUT_OF_CONTEXT_RESPONSE,
                context_used="", is_grounded=False,
            )
 
        # Step 1 — scope guard via LangChain chain
        if not skip_scope_check:
            if not self._check_scope(query):
                return AnswerResult(
                    query=query, answer=OUT_OF_SCOPE_RESPONSE,
                    context_used="", is_out_of_scope=True, is_grounded=False,
                )
 
        # Step 2 — answer via LangChain chain
        answer = self._generate_answer(query, context)
        is_grounded = OUT_OF_CONTEXT_RESPONSE.lower() not in answer.lower()
 
        return AnswerResult(
            query=query, answer=answer,
            context_used=context, is_grounded=is_grounded,
        )
 
    def summarise(self, context: str) -> AnswerResult:
        if not context.strip():
            raise ValueError("Context cannot be empty.")
 
        prompt_text = prompt_manager.load(
            "summarization_prompt", variables={"context": context}
        )
        # Invoke LangChain summary chain
        summary = self._summary_chain.invoke({"summary_prompt": prompt_text})
 
        return AnswerResult(
            query="Summarise the contract.",
            answer=summary.strip(),
            context_used=context,
            is_grounded=True,
        )
 
    # ── Private helpers ────────────────────────────────────────────────
    def _check_scope(self, query: str) -> bool:
        try:
            prompt_text = prompt_manager.load(
                "scope_guard_prompt", variables={"question": query}
            )
            # Invoke LangChain scope chain
            result = self._scope_chain.invoke({"scope_prompt": prompt_text})
            return "IN_SCOPE" in result.strip().upper()
        except Exception as exc:
            logger.warning("Scope check failed — defaulting IN_SCOPE: %s", exc)
            return True
 
    def _generate_answer(self, query: str, context: str) -> str:
        prompt_text = prompt_manager.load(
            "answer_prompt",
            variables={"context": context, "question": query},
        )
        # Invoke LangChain answer chain
        answer = self._answer_chain.invoke({"answer_prompt": prompt_text})
        return answer.strip()
 
 
answer_generator = AnswerGenerator()