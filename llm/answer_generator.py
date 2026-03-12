"""
llm/answer_generator.py
------------------------
LangChain 1.x compatible answer generator.
Uses AzureChatOpenAI + ChatPromptTemplate + StrOutputParser chains.
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
    LangChain 1.x answer generator using pipe chains:
      scope_chain   → IN_SCOPE / OUT_OF_SCOPE classifier
      answer_chain  → grounded contract Q&A
      summary_chain → executive contract summary
    """
 
    def __init__(self) -> None:
        cfg    = settings.azure_openai
        parser = StrOutputParser()
 
        # Main LLM
        llm = AzureChatOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_deployment=cfg.chat_deployment,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            temperature=0,
            max_tokens=1500,
        )
 
        # Short LLM for scope check (only needs 10 tokens)
        llm_short = AzureChatOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_deployment=cfg.chat_deployment,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            temperature=0,
            max_tokens=10,
        )
 
        # ── Chain 1: scope guard ────────────────────────────────────────
        self._scope_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a query classifier. Respond with only IN_SCOPE or OUT_OF_SCOPE."),
                ("human",  "{input}"),
            ])
            | llm_short
            | parser
        )
 
        # ── Chain 2: answer ─────────────────────────────────────────────
        self._answer_chain = (
            ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a precise contract analysis assistant. "
                    "Answer questions strictly based on the provided context. "
                    f"If the answer is not in the context, respond exactly: '{OUT_OF_CONTEXT_RESPONSE}'",
                ),
                ("human", "{input}"),
            ])
            | llm
            | parser
        )
 
        # ── Chain 3: summary ────────────────────────────────────────────
        self._summary_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a senior legal analyst. Produce structured contract summaries in clear business language."),
                ("human",  "{input}"),
            ])
            | llm
            | parser
        )
 
        logger.info("AnswerGenerator initialised (LangChain 1.x).")
 
    # ── Public API ─────────────────────────────────────────────────────
    def generate(self, query: str, context: str, skip_scope_check: bool = False) -> AnswerResult:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
 
        if not context.strip():
            return AnswerResult(
                query=query, answer=OUT_OF_CONTEXT_RESPONSE,
                context_used="", is_grounded=False,
            )
 
        if not skip_scope_check and not self._check_scope(query):
            return AnswerResult(
                query=query, answer=OUT_OF_SCOPE_RESPONSE,
                context_used="", is_out_of_scope=True, is_grounded=False,
            )
 
        answer      = self._generate_answer(query, context)
        is_grounded = OUT_OF_CONTEXT_RESPONSE.lower() not in answer.lower()
 
        return AnswerResult(
            query=query, answer=answer,
            context_used=context, is_grounded=is_grounded,
        )
 
    def summarise(self, context: str) -> AnswerResult:
        prompt_text = prompt_manager.load("summarization_prompt", variables={"context": context})
        summary     = self._summary_chain.invoke({"input": prompt_text})
        return AnswerResult(
            query="Summarise the contract.", answer=summary.strip(),
            context_used=context, is_grounded=True,
        )
 
    # ── Private ────────────────────────────────────────────────────────
    def _check_scope(self, query: str) -> bool:
        try:
            prompt_text = prompt_manager.load("scope_guard_prompt", variables={"question": query})
            result      = self._scope_chain.invoke({"input": prompt_text})
            return "IN_SCOPE" in result.strip().upper()
        except Exception as exc:
            logger.warning("Scope check failed — defaulting IN_SCOPE: %s", exc)
            return True
 
    def _generate_answer(self, query: str, context: str) -> str:
        prompt_text = prompt_manager.load(
            "answer_prompt", variables={"context": context, "question": query}
        )
        return self._answer_chain.invoke({"input": prompt_text}).strip()
 
 
answer_generator = AnswerGenerator()
 