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
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        prompt_path = Path(cls.PROMPTS_DIR) / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')

config = Config()

