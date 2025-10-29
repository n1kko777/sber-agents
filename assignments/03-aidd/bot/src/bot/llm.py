from typing import Iterable

from openai import AsyncOpenAI

from .config import Settings

SYSTEM_PROMPT = (
    "You are an English language teacher. "
    "Explain concepts clearly, provide gentle corrections, and encourage practice."
)


class TutorLLM:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = settings.openrouter_model

    async def reply(self, user_message: str) -> str:
        response = await self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return self._extract_text(response.output) or "I'm still thinking about that."

    @staticmethod
    def _extract_text(chunks: Iterable) -> str:
        for chunk in chunks or []:
            for part in getattr(chunk, "content", []) or []:
                if getattr(part, "type", None) == "text":
                    return part.text
        return ""
