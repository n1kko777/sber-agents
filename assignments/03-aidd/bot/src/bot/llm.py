from typing import Iterable, List, Tuple

from openai import AsyncOpenAI

from .config import Settings

SYSTEM_PROMPT = (
    "You are an English language teacher. "
    "Explain concepts clearly, provide gentle corrections, and encourage practice."
)


Message = Tuple[str, str]


class TutorLLM:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = settings.openrouter_model

    async def reply(self, user_message: str, history: List[Message]) -> str:
        payload = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in history:
            payload.append({"role": role, "content": text})
        payload.append({"role": "user", "content": user_message})

        response = await self._client.responses.create(
            model=self._model,
            input=payload,
        )
        return self._extract_text(response.output) or "I'm still thinking about that."

    @staticmethod
    def _extract_text(chunks: Iterable) -> str:
        for chunk in chunks or []:
            for part in getattr(chunk, "content", []) or []:
                if getattr(part, "type", None) == "text":
                    return part.text
        return ""
