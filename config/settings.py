"""
config/settings.py
------------------
Central configuration module for the Contract Intelligence System.
Loads all environment variables, validates required fields,
and exposes a typed settings singleton used across the application.
 
Usage:
    from config.settings import settings
    print(settings.azure_openai.endpoint)
"""
 
import logging
import os
from dataclasses import dataclass
from pathlib import Path
 
from dotenv import load_dotenv
 
# ---------------------------------------------------------------------------
# Resolve project root and load .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file at: {PROJECT_ROOT / '.env'}"
        )
    return value
 
 
def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)
 
 
# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str
 
 
@dataclass(frozen=True)
class AzureSearchConfig:
    endpoint: str
    api_key: str
    index_name: str
 
 
@dataclass(frozen=True)
class AzureDocumentIntelligenceConfig:
    endpoint: str
    api_key: str
 
 
@dataclass(frozen=True)
class CosmosDBConfig:
    gremlin_endpoint: str
    primary_key: str
    database_name: str
    graph_name: str
 
 
@dataclass(frozen=True)
class AppConfig:
    retrieval_top_k: int
    memory_window_size: int
    chunk_size: int
    chunk_overlap: int
    log_level: str
    prompts_dir: Path
    project_root: Path
 
 
@dataclass(frozen=True)
class Settings:
    azure_openai: AzureOpenAIConfig
    azure_search: AzureSearchConfig
    document_intelligence: AzureDocumentIntelligenceConfig
    cosmos_db: CosmosDBConfig
    app: AppConfig
 
 
# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
def _build_settings() -> Settings:
    azure_openai = AzureOpenAIConfig(
        endpoint=_require("AZURE_OPENAI_ENDPOINT").rstrip("/"),
        api_key=_require("AZURE_OPENAI_API_KEY"),
        api_version=_optional("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        chat_deployment=_require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        embedding_deployment=_require("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    azure_search = AzureSearchConfig(
        endpoint=_require("AZURE_SEARCH_ENDPOINT").rstrip("/"),
        api_key=_require("AZURE_SEARCH_API_KEY"),
        index_name=_optional("AZURE_SEARCH_INDEX_NAME", "contract-chunks"),
    )
    document_intelligence = AzureDocumentIntelligenceConfig(
        endpoint=_require("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT").rstrip("/"),
        api_key=_require("AZURE_DOCUMENT_INTELLIGENCE_API_KEY"),
    )
    cosmos_db = CosmosDBConfig(
        gremlin_endpoint=_require("COSMOS_GREMLIN_ENDPOINT"),
        primary_key=_require("COSMOS_PRIMARY_KEY"),
        database_name=_optional("COSMOS_DATABASE_NAME", "ContractGraph"),
        graph_name=_optional("COSMOS_GRAPH_NAME", "contracts"),
    )
    app = AppConfig(
        retrieval_top_k=int(_optional("RETRIEVAL_TOP_K", "5")),
        memory_window_size=int(_optional("MEMORY_WINDOW_SIZE", "5")),
        chunk_size=int(_optional("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_optional("CHUNK_OVERLAP", "200")),
        log_level=_optional("LOG_LEVEL", "INFO").upper(),
        prompts_dir=PROJECT_ROOT / "prompts",
        project_root=PROJECT_ROOT,
    )
    logger.info("Settings loaded successfully from: %s", PROJECT_ROOT / ".env")
    return Settings(
        azure_openai=azure_openai,
        azure_search=azure_search,
        document_intelligence=document_intelligence,
        cosmos_db=cosmos_db,
        app=app,
    )
 
 
# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
settings: Settings = _build_settings()
 
 
# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Contract Intelligence — Configuration Check ===\n")
    print(f"  Project Root           : {settings.app.project_root}")
    print(f"  Prompts Directory      : {settings.app.prompts_dir}")
    print(f"  OpenAI Endpoint        : {settings.azure_openai.endpoint}")
    print(f"  Chat Deployment        : {settings.azure_openai.chat_deployment}")
    print(f"  Embedding Deployment   : {settings.azure_openai.embedding_deployment}")
    print(f"  Search Endpoint        : {settings.azure_search.endpoint}")
    print(f"  Search Index           : {settings.azure_search.index_name}")
    print(f"  Doc Intelligence       : {settings.document_intelligence.endpoint}")
    print(f"  Cosmos Endpoint        : {settings.cosmos_db.gremlin_endpoint}")
    print(f"  Cosmos Database        : {settings.cosmos_db.database_name}")
    print(f"  Cosmos Graph           : {settings.cosmos_db.graph_name}")
    print(f"  Retrieval Top-K        : {settings.app.retrieval_top_k}")
    print(f"  Memory Window          : {settings.app.memory_window_size}")
    print(f"  Chunk Size             : {settings.app.chunk_size}")
    print(f"  Chunk Overlap          : {settings.app.chunk_overlap}")
    print(f"  Log Level              : {settings.app.log_level}")
    print("\n  All settings validated successfully.\n")
 
