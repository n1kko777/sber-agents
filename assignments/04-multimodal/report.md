## Отчёт о выполнении задания

- Проект: Персональный финансовый советник (Telegram-бот)
- Краткое описание: Бот ведёт учёт доходов и расходов из сообщений в чате, извлекает транзакции из текста, изображений чеков и голосовых сообщений, формирует баланс и простую статистику. Поддерживает облачные модели через OpenRouter и локальные через Ollama.

## Вариант задания

Расширенный (текст + изображения + голос, отчёты, локальные модели через Ollama).

## Реализованные возможности

- [x] Извлечение транзакций из текстовых сообщений (structured output через Pydantic)
- [x] Обработка изображений (чеки/скриншоты) с VLM и structured output
- [x] Транскрибация голосовых (OpenAI Whisper API или локально faster-whisper + ffmpeg)
- [x] Автоматическая категоризация (продукты, рестораны, такси, …; возможны новые категории)
- [x] Отчёт о балансе и статистике по категориям (`/balance`)
- [x] История всех транзакций с сортировкой (`/transactions`)
- [x] Поддержка локальных моделей через Ollama (единый клиент OpenAI с кастомным `base_url`)
- [x] Удаление последней транзакции по фразе (“удали последнюю …”) и явные шаблоны доходов (зарплата)
- [x] Обработка ошибок провайдера и понятные сообщения пользователю
- [x] Ограничение длины сообщения и простое хранение в памяти (без БД)

## Технологический стек

- Python 3.11+, uv (менеджер зависимостей)
- Telegram: `aiogram` 3.x (polling, Router)
- LLM-клиент: `openai>=1.54` (AsyncOpenAI) с `base_url` для OpenRouter/Ollama
- Валидация/схемы: `pydantic` v2 (JSON Schema для structured output)
- Конфигурация: `python-dotenv`
- STT: OpenAI Whisper API (облако) или `faster-whisper` (локально); `ffmpeg` для конвертации OGG/Opus → WAV 16kHz mono
- Используемые модели (из `.env`/README):
  - OpenRouter (облако): `openai/gpt-oss-20b:free` (текст), `meta-llama/llama-3.2-11b-vision-instruct` (изображения)
  - Ollama (локально): `llama3.2` (текст), `llama3.2-vision` (изображения)
  - STT: `openai/whisper-1` (через OpenRouter или напрямую), локально — `faster-whisper` (`small`/`medium`/`large-v3` и т.д.)

## Инструменты AI‑driven разработки

- Codex CLI (агент-помощник в терминале) для ускорения разработки и документирования
- LLM-провайдер: OpenRouter (единый API для опенсорс‑моделей); локально — Ollama

## Скриншоты работы

- [Старт + голосовое](screenshots/start_message_and_audio.png)
- [Распознавание чека/изображения](screenshots/image_instract_text.png)
- [Транскрибация голосового и извлечение транзакций](screenshots/voice_operation_transcribe.png)

## Облачный сервер / окружение

- Провайдер LLM (облако): OpenRouter (`https://openrouter.ai/api/v1`)
- Развёртывание бота: локально (polling) — отдельный сервер не обязателен
- GPU: не требуется для облачных вызовов; опционален локально (ускорение `faster-whisper` на `cuda`)
- Локальные модели Ollama: `llama3.2`, `llama3.2-vision` (`ollama pull …`), запуск `ollama serve`

## Основные вызовы и решения

- Единый LLM‑клиент: один `AsyncOpenAI` с переключением `base_url` и моделей для OpenRouter/Ollama
- Надёжный structured output: JSON Schema из Pydantic, подробный лог “сырых” ответов и разбор с защитой от отсутствующих полей
- Vision‑обработка: передача изображений как `image_url` (data URL, base64), проверка поддержки vision‑модели и дружелюбные ошибки
- Голос: неблокирующая конвертация через `ffmpeg` в отдельном потоке + провайдеры STT (OpenAI/`faster-whisper`) и обработка 405 у OpenRouter
- Простая архитектура: in‑memory хранилище диалогов и транзакций, KISS/YAGNI

## Что узнал нового

- Практика structured output: `response_format` с JSON Schema из Pydantic v2
- Использование `openai` клиента с кастомным `base_url` (OpenRouter/Ollama)
- Отправка изображений в chat.completions через content‑part `image_url`
- Интеграция `faster-whisper` + `ffmpeg` и выбор compute‑профиля (CPU/GPU)
- Паттерны aiogram 3 (Router, обработчики для photo/voice/document)

---

Дополнительно см.: README.md, docs/vision.md, docs/tasklist.md, prompts/
