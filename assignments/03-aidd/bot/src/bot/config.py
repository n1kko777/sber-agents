import os
from dataclasses import dataclass
from typing import Dict


TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


@dataclass
class Settings:
    telegram_token: str
    openrouter_key: str
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from environment variables."""
        telegram_token = os.environ[TELEGRAM_TOKEN_ENV]
        openrouter_key = os.environ[OPENROUTER_KEY_ENV]
        openrouter_model = os.getenv(OPENROUTER_MODEL_ENV, DEFAULT_OPENROUTER_MODEL)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        return cls(
            telegram_token=telegram_token,
            openrouter_key=openrouter_key,
            openrouter_model=openrouter_model,
            log_level=log_level,
        )

    def public_info(self) -> Dict[str, str]:
        return {
            "openrouter_model": self.openrouter_model,
            "log_level": self.log_level,
        }
