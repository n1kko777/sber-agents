import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("MODEL")
    MODEL_QUERY_TRANSFORM = os.getenv("MODEL_QUERY_TRANSFORM", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
    CONVERSATION_SYSTEM_PROMPT_FILE = os.getenv("CONVERSATION_SYSTEM_PROMPT_FILE", "conversation_system.txt")
    QUERY_TRANSFORM_PROMPT_FILE = os.getenv("QUERY_TRANSFORM_PROMPT_FILE", "query_transform.txt")
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # Отображение источников
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    
    # LangSmith настройки
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # RAGAS evaluation настройки (фиксированные модели для единообразной оценки)
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o")
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        prompt_path = Path(cls.PROMPTS_DIR) / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')

config = Config()

