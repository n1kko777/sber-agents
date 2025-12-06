import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
    AGENT_SYSTEM_PROMPT_FILE = os.getenv("AGENT_SYSTEM_PROMPT_FILE", "agent_system.txt")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai/huggingface
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    HUGGINGFACE_DEVICE = os.getenv("HUGGINGFACE_DEVICE", "cpu")  # cpu/cuda/mps
    
    # Retrieval Configuration
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "semantic")  # semantic/hybrid/hybrid_reranker
    SEMANTIC_RETRIEVER_K = int(os.getenv("SEMANTIC_RETRIEVER_K", "10"))
    BM25_RETRIEVER_K = int(os.getenv("BM25_RETRIEVER_K", "10"))
    ENSEMBLE_SEMANTIC_WEIGHT = float(os.getenv("ENSEMBLE_SEMANTIC_WEIGHT", "0.5"))
    ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.5"))
    
    # Cross-Encoder Reranking Configuration
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    
    # Отображение источников
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    
    # MCP Configuration
    MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"
    MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "mcp-bank-agent")
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    MCP_SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "streamable_http")
    
    # LangSmith настройки
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    # Поддержка обеих переменных для совместимости (стандартная - LANGSMITH_TRACING_V2)
    _tracing = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGSMITH_TRACING") or "false"
    LANGSMITH_TRACING_V2 = _tracing.lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # RAGAS evaluation настройки (фиксированные модели для единообразной оценки)
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o")
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)  # По умолчанию = основному провайдеру
    # Для HuggingFace используем те же настройки что и для основных embeddings
    RAGAS_HUGGINGFACE_EMBEDDING_MODEL = os.getenv("RAGAS_HUGGINGFACE_EMBEDDING_MODEL", HUGGINGFACE_EMBEDDING_MODEL)
    RAGAS_HUGGINGFACE_DEVICE = os.getenv("RAGAS_HUGGINGFACE_DEVICE", HUGGINGFACE_DEVICE)
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        prompt_path = Path(cls.PROMPTS_DIR) / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')
    
    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        # Валидация RETRIEVAL_MODE
        valid_retrieval_modes = ["semantic", "hybrid", "hybrid_reranker"]
        if cls.RETRIEVAL_MODE not in valid_retrieval_modes:
            raise ValueError(
                f"Invalid RETRIEVAL_MODE: {cls.RETRIEVAL_MODE}. "
                f"Must be one of: {', '.join(valid_retrieval_modes)}"
            )
        
        # Валидация EMBEDDING_PROVIDER
        valid_embedding_providers = ["openai", "huggingface"]
        if cls.EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {cls.EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )
        
        # Валидация RAGAS_EMBEDDING_PROVIDER
        if cls.RAGAS_EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid RAGAS_EMBEDDING_PROVIDER: {cls.RAGAS_EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )

config = Config()
# Валидация конфигурации при загрузке
config.validate()

