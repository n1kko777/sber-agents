import asyncio
import os
import shutil
import subprocess
from typing import Optional

from config import config


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg не найден. Установите ffmpeg (macOS: brew install ffmpeg, Ubuntu/Debian: apt-get install ffmpeg)."
        )
    return path


async def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> None:
    ffmpeg = _ensure_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    # Выполняем преобразование в отдельном потоке, чтобы не блокировать event loop
    def _run():
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await asyncio.to_thread(_run)


async def transcribe_audio(path_wav: str) -> str:
    provider = (config.STT_PROVIDER or "openai").lower()
    if provider == "openai":
        return await _transcribe_openai(path_wav)
    elif provider in {"faster_whisper", "faster-whisper", "local"}:
        return await _transcribe_faster_whisper(path_wav)
    else:
        raise ValueError(f"Неизвестный STT_PROVIDER: {config.STT_PROVIDER}")


async def _transcribe_openai(path_wav: str) -> str:
    # Локальный клиент с отдельной базой/ключом для STT (может отличаться от чата)
    from openai import AsyncOpenAI, APIStatusError

    client = AsyncOpenAI(
        api_key=config.OPENAI_STT_API_KEY,
        base_url=config.OPENAI_AUDIO_BASE_URL,
    )
    model = config.STT_OPENAI_MODEL or "whisper-1"

    def _call():
        with open(path_wav, "rb") as f:
            return f.read()

    data = await asyncio.to_thread(_call)

    # Оборачиваем в BytesIO для передачи без повторного открытия
    import io

    fileobj = io.BytesIO(data)
    fileobj.name = os.path.basename(path_wav)

    try:
        resp = await client.audio.transcriptions.create(
            model=model,
            file=fileobj,
            language=config.STT_LANGUAGE or "ru",
        )
        return (resp.text or "").strip()
    except APIStatusError as e:
        # Частый случай: OpenRouter может вернуть 405 для audio.transcriptions
        msg = str(e)
        if getattr(e, "status_code", None) == 405 or "405" in msg or "method not allowed" in msg.lower():
            raise RuntimeError(
                "Провайдер OpenRouter не поддерживает endpoint audio.transcriptions для вашей конфигурации. "
                "Укажите OPENAI_AUDIO_BASE_URL=https://api.openai.com/v1 и STT_OPENAI_MODEL=whisper-1, либо переключите STT_PROVIDER=faster_whisper."
            ) from e
        raise


_fw_model = None


async def _transcribe_faster_whisper(path_wav: str) -> str:
    global _fw_model
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Требуется пакет 'faster-whisper'. Установите: uv add 'faster-whisper>=1.0.0'"
        ) from e

    def _ensure_model():
        global _fw_model
        if _fw_model is None:
            _fw_model = WhisperModel(
                config.STT_LOCAL_MODEL or "small",
                device=config.STT_DEVICE or "cpu",
                compute_type=config.STT_COMPUTE_TYPE or "int8",
            )
        return _fw_model

    model = await asyncio.to_thread(_ensure_model)

    def _run_transcribe():
        vad = str(config.STT_VAD or "false").lower() in ("1", "true", "yes", "on")
        segments, _info = model.transcribe(
            path_wav,
            language=config.STT_LANGUAGE or "ru",
            vad_filter=vad,
            beam_size=5,
        )
        return "".join(seg.text for seg in segments).strip()

    return await asyncio.to_thread(_run_transcribe)
