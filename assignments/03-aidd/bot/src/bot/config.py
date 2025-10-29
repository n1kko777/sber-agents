import os
from dataclasses import dataclass


TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
DEFAULT_OPENROUTER_MODEL = "openrouter/anthropic/claude-3.5-sonnet"


@dataclass
class Settings:
    telegram_token: str
    openrouter_key: str
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from environment variables."""
        telegram_token = os.environ[TELEGRAM_TOKEN_ENV]
        openrouter_key = os.environ[OPENROUTER_KEY_ENV]
        openrouter_model = os.getenv(OPENROUTER_MODEL_ENV, DEFAULT_OPENROUTER_MODEL)
        return cls(
            telegram_token=telegram_token,
            openrouter_key=openrouter_key,
            openrouter_model=openrouter_model,
        )
