from typing import Iterable, List, Tuple

from openai import AsyncOpenAI

from .config import Settings

SYSTEM_PROMPT = (
    "You are an English language teacher. "
    "Explain concepts clearly, provide gentle corrections, and encourage practice. "
    "Always answer in plain text without Markdown tables or headings; "
    "use short paragraphs and simple bullet lists with hyphens when needed."
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
        text = getattr(response, "output_text", None)
        if text:
            return self._to_plain_text(text)
        raw = self._extract_text(response.output)
        return self._to_plain_text(raw) if raw else "I'm still thinking about that."

    @staticmethod
    def _extract_text(chunks: Iterable) -> str:
        for chunk in chunks or []:
            for part in getattr(chunk, "content", []) or []:
                if getattr(part, "type", None) == "text":
                    return part.text
        return ""

    @staticmethod
    def _to_plain_text(text: str) -> str:
        if not text:
            return ""
        cleaned = (
            text.replace("**", "")
            .replace("__", "")
            .replace("```", "")
            .replace("`", "")
        )
        lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if stripped.startswith("#"):
                stripped = stripped.lstrip("# ").strip()
            if "|" in stripped and stripped.count("|") >= 2:
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                stripped = " | ".join(cell for cell in cells if cell)
            lines.append(stripped)
        return "\n".join(lines).strip()
