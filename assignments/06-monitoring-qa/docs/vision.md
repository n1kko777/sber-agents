# Техническое видение проекта

## Технологии

**Основные технологии:**
- **Python 3.11+** - основной язык разработки
- **uv** - управление зависимостями и виртуальным окружением
- **aiogram 3.x** - фреймворк для Telegram Bot API (polling)
- **LangChain** - фреймворк для построения RAG-приложений
- **langchain-openai** - интеграция LangChain с OpenAI-совместимыми API
- **openai** - клиент для работы с LLM через Openrouter
- **pypdf** - загрузка и парсинг PDF-документов
- **python-dotenv** - для работы с переменными окружения
- **Make** - автоматизация сборки и запуска

## Принципы разработки

**Принципы:**
- **KISS** (Keep It Simple, Stupid) - максимальная простота решений
- **YAGNI** (You Aren't Gonna Need It) - реализуем только то, что нужно сейчас
- **Монолитная архитектура** - весь код в одном месте, никаких микросервисов
- **Прямолинейный код** - минимум абстракций, максимум читаемости
- **Быстрый старт** - от идеи до рабочего прототипа за минимальное время

**Что НЕ делаем:**
- Не создаем сложные архитектурные паттерны
- Не делаем преждевременную оптимизацию
- Не добавляем функции "на будущее"
- Не усложняем без крайней необходимости

## Структура проекта

```
/
├── src/
│   ├── bot.py                  # Основной файл бота, инициализация aiogram
│   ├── handlers.py             # Обработчики команд и сообщений Telegram
│   ├── rag.py                  # RAG-логика: retriever, цепочки, query transformation
│   ├── indexer.py              # Индексация: загрузка PDF, splitting, векторное хранилище
│   ├── config.py               # Загрузка конфигурации из .env
│   ├── evaluation.py           # Оценка качества RAG через RAGAS (новое)
│   └── dataset_synthesizer.py  # Синтез тестовых датасетов (новое)
├── data/               # Директория с PDF-документами для индексации
├── datasets/           # Синтезированные тестовые датасеты (новое)
├── prompts/            # Промпты для RAG и синтеза датасетов
├── logs/               # Логи работы бота
├── .env                # Переменные окружения (токены, настройки)
├── .env.example        # Пример конфигурации
├── pyproject.toml      # Конфигурация проекта для uv
├── Makefile            # Команды для запуска и управления
└── README.md           # Документация по запуску
```

**Принцип:** Простая структура - все Python-файлы в одной папке `src/`. Никаких пакетов, подпакетов, сложной иерархии.

## Архитектура проекта

**Компоненты:**

1. **bot.py** - точка входа
   - Инициализирует aiogram Bot и Dispatcher
   - Запускает индексацию документов при старте
   - Регистрирует handlers
   - Запускает polling

2. **handlers.py** - обработка событий
   - `/start` - приветствие и очистка истории
   - `/help` - справка по командам и возможностям
   - `/index` - ручная переиндексация документов
   - `/index_status` - статус индексации
   - Обработчик всех текстовых сообщений → вызов RAG → сохранение ответа в историю
   - Хранит историю диалогов в памяти: `dict[int, list]` (chat_id → список сообщений)

3. **indexer.py** - индексация документов
   - `load_pdf_documents(data_dir)` - загрузка PDF через PyPDFLoader
   - `split_documents(pages)` - разбиение на чанки через RecursiveCharacterTextSplitter
   - `create_vector_store(chunks)` - создание InMemoryVectorStore с эмбеддингами
   - `reindex_all()` - полная переиндексация с нуля
   - Глобальная переменная `vector_store` для хранения векторного хранилища

4. **rag.py** - RAG-логика
   - `format_chunks(chunks)` - форматирование чанков в строку
   - `retrieval_query_transformation_chain` - цепочка трансформации запроса
   - `rag_query_transform_chain` - финальная RAG-цепочка
   - `rag_answer(messages)` - асинхронный метод для получения RAG-ответа
   - Промпты для conversation и query transformation
   - Использует retriever из vector_store

5. **llm.py** - интеграция с LLM
   - `get_response(message_history: list) -> str` - базовый метод для LLM
   - Отправляет запрос в OpenRouter через openai client
   - Используется для простых запросов без RAG

6. **config.py** - конфигурация
   - Класс Config с полями: `TELEGRAM_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL`, `MODEL_QUERY_TRANSFORM`, `EMBEDDING_MODEL`, `DATA_DIR`, `SYSTEM_PROMPT`
   - Новые поля для мониторинга: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING_V2`, `LANGSMITH_PROJECT`, `LANGSMITH_DATASET`
   - Флаг отображения источников: `SHOW_SOURCES`
   - Загрузка из .env через python-dotenv

7. **evaluation.py** - оценка качества RAG (новое)
   - `evaluate_dataset(dataset_name)` - запуск evaluation на датасете из LangSmith
   - `check_dataset_exists(name)` - проверка существования датасета
   - `evaluate_with_ragas()` - batch вычисление RAGAS метрик
   - `upload_feedback()` - загрузка результатов в LangSmith
   - Метрики: faithfulness, answer_relevancy, answer_correctness, answer_similarity
   - Используется LangSmith API и RAGAS библиотека

8. **dataset_synthesizer.py** - синтез датасетов (новое)
   - `synthesize_dataset()` - создание QA пар из документов
   - `load_and_sample_documents()` - выборка чанков (по 2 на файл)
   - `synthesize_qa_pairs()` - генерация вопросов и ответов через LLM
   - `upload_to_langsmith()` - загрузка датасета в LangSmith с проверкой дубликатов
   - Запуск через CLI: `python -m src.dataset_synthesizer [--upload]`

**Поток данных (RAG):**
```
Telegram → handlers.py (добавить в историю) → 
rag.py (query transformation → retriever → context augmentation) → 
LLM → rag.py (возврат answer + documents) → 
handlers.py (сохранить ответ, форматировать sources если SHOW_SOURCES=true) → Telegram
```

**Поток данных (Evaluation):**
```
Telegram `/evaluate-dataset` → handlers.py → evaluation.py:

1. Загрузка датасета из LangSmith API
   ↓
2. Запуск RAG для каждого вопроса с трейсингом в LangSmith
   (одновременно собираем: questions, answers, contexts, ground_truths, run_ids)
   ↓
3. Batch вычисление RAGAS метрик на всех собранных данных
   (faithfulness, answer_relevancy, answer_correctness, answer_similarity)
   ↓
4. Загрузка результатов как feedback в LangSmith 
   (привязка метрик к соответствующим run_ids)
   ↓
5. Возврат агрегированных результатов → handlers.py → Telegram

Гибридный подход: трейсинг в LangSmith + эффективный batch processing RAGAS
```

**Принцип:** Никакой DI, никаких интерфейсов, никаких слоев абстракции. Прямые вызовы функций. Глобальные переменные для простых хранилищ.

## Модель данных

**Хранение в памяти (без БД):**

Глобальный словарь в `handlers.py`:
```python
chat_conversations: dict[int, list[dict]] = {}
```

**Структура истории диалога:**
```python
chat_conversations[chat_id] = [
    {"role": "system", "content": "системный промпт"},
    {"role": "user", "content": "сообщение пользователя"},
    {"role": "assistant", "content": "ответ LLM"},
    ...
]
```

**Операции:**
- При `/start` - очищаем историю для данного чата
- При новом сообщении - добавляем в список
- Передаем весь список в LLM для контекста
- При перезапуске бота - вся история теряется

**Принцип:** Максимальная простота. Никаких БД, файлов, сериализации. История живет только в runtime.

## Работа с LLM

**Используемая библиотека:** `openai` (официальный Python client, асинхронная версия)

**Настройка:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL  # https://openrouter.ai/api/v1
)
```

**Основной метод в llm.py:**
```python
async def get_response(message_history: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=config.MODEL,  # например "openai/gpt-oss-20b:free"
        messages=message_history
    )
    return response.choices[0].message.content
```

**Параметры из .env:**
- `OPENAI_API_KEY` - ключ от OpenRouter
- `OPENAI_BASE_URL` - `https://openrouter.ai/api/v1`
- `MODEL` - название модели (например `openai/gpt-oss-20b:free`)
- `SYSTEM_PROMPT` - роль/инструкция для LLM

**Обработка ошибок:**
- try/except для сетевых ошибок
- Возврат простого сообщения об ошибке пользователю

**Принцип:** Асинхронный запрос-ответ. Никакого retry, никаких очередей, никакого streaming.

## Сценарии работы

**Сценарий 1: Первый запуск**
1. Пользователь отправляет `/start`
2. Бот отвечает приветственным сообщением
3. История диалога инициализируется с системным промптом

**Сценарий 2: Диалог**
1. Пользователь пишет текстовое сообщение
2. Бот добавляет сообщение в историю чата
3. Бот отправляет историю чата в LLM
4. Бот получает ответ и добавляет его в историю чата
5. Бот отправляет ответ пользователю

**Сценарий 3: Сброс контекста**
1. Пользователь отправляет `/start`
2. История диалога очищается
3. Начинается новый диалог

**Ограничения:**
- Бот работает только с текстом (не обрабатывает фото, файлы, голосовые)
- Один пользователь не блокирует других (асинхронность)
- При перезапуске бота все истории теряются

## Подход к конфигурированию

**Файл .env** (не коммитится в git):
```bash
# Telegram Bot
TELEGRAM_TOKEN=your_telegram_bot_token

# LLM Provider
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
MODEL_QUERY_TRANSFORM=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Data & Prompts
DATA_DIR=data
SYSTEM_PROMPT=Ты ассистент Сбербанка, отвечающий на вопросы по документам.

# LangSmith (опционально, для трейсинга и evaluation)
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=rag-bot
LANGSMITH_DATASET=06-rag-qa-dataset

# Features
SHOW_SOURCES=false
```

**Файл .env.example** (коммитится):
```bash
# Telegram Bot
TELEGRAM_TOKEN=

# LLM Provider
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
MODEL_QUERY_TRANSFORM=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Data & Prompts
DATA_DIR=data
SYSTEM_PROMPT=Ты ассистент, отвечающий на вопросы по документам.

# LangSmith (опционально, для трейсинга и evaluation)
LANGSMITH_API_KEY=
LANGSMITH_TRACING_V2=false
LANGSMITH_PROJECT=rag-bot
LANGSMITH_DATASET=06-rag-qa-dataset

# Features
SHOW_SOURCES=false
```

**config.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram & LLM
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("MODEL")
    MODEL_QUERY_TRANSFORM = os.getenv("MODEL_QUERY_TRANSFORM")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # LangSmith (опционально)
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "false")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-bot")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # Features
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"

config = Config()
```

**Принципы:**
- Все секреты только в .env
- Нет YAML, JSON, TOML конфигов
- Нет окружений (dev/prod)
- Нет валидации на старте (упадет при первом использовании если что-то не так)

## Подход к логгированию

**Используем встроенный logging Python:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Что логируем:**
- Старт/остановка бота
- Входящие сообщения от пользователей (chat_id + текст)
- Ошибки при вызове LLM
- Исключения

**Что НЕ логируем:**
- Содержимое ответов LLM (избыточно для MVP)
- Детальные трейсы успешных операций
- Метрики, аналитика

**Вывод:** Только в stdout/stderr (консоль)

**Принципы:**
- Без внешних библиотек (structlog и т.п.)
- Без файлов, ротации логов
- Без отправки в внешние системы
- Простой текстовый формат

## Оценка качества RAG (RAGAS)

### Метрики

**faithfulness** (обоснованность):
- Проверяет, что ответ основан на retrieved documents
- Отсутствие галлюцинаций и выдуманных фактов
- Значение от 0.0 до 1.0 (выше = лучше)

**answer_relevancy** (релевантность):
- Насколько ответ релевантен заданному вопросу
- Проверяет, что ответ действительно отвечает на вопрос
- Значение от 0.0 до 1.0 (выше = лучше)

**answer_correctness** (правильность):
- Правильность ответа относительно ground truth (эталона)
- Комбинирует факт-чекинг и семантическую похожесть
- Значение от 0.0 до 1.0 (выше = лучше)

**answer_similarity** (семантическая похожесть):
- Семантическая похожесть ответа на ground truth
- Фокус на смысле, а не на точных формулировках
- Значение от 0.0 до 1.0 (выше = лучше)

### Интеграция с LangSmith

**Автоматический трейсинг:**
Все вызовы RAG цепочки автоматически логируются в LangSmith при установке переменных окружения:
- `LANGSMITH_TRACING_V2=true` - включает трейсинг
- `LANGSMITH_PROJECT=<name>` - группирует traces в проект
- Детальные traces: latency, tokens, промежуточные шаги

**Датасеты:**
- Хранение тестовых QA пар в LangSmith
- Формат: inputs (question), outputs (answer), metadata (contexts)
- API для загрузки и получения датасетов

**Feedback:**
- Результаты RAGAS метрик загружаются как feedback к traces
- Визуализация в LangSmith UI
- Сравнение экспериментов и версий RAG pipeline

### Синтез датасетов

**Процесс:**
1. Загрузка PDF документов из data/
2. Выбор репрезентативных чанков (по 2 на файл)
3. Генерация QA пар через LLM с промптом из ноутбука
4. Сохранение в JSON: datasets/06-rag-qa-dataset.json
5. Загрузка в LangSmith с проверкой дубликатов

**Формат датасета:**
```json
{
  "question": "Вопрос по документу",
  "ground_truth": "Правильный ответ",
  "contexts": ["Релевантный текст из документа"],
  "metadata": {
    "source": "filename.pdf",
    "page": 3
  }
}
```


