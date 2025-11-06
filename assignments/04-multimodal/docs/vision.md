# Техническое видение проекта

## Технологии

**Основные технологии:**
- **Python 3.11+** - основной язык разработки
- **uv** - управление зависимостями и виртуальным окружением
- **aiogram 3.x** - фреймворк для Telegram Bot API (polling)
- **openai** - клиент для работы с LLM через OpenRouter/Ollama (единый интерфейс)
- **pydantic** - валидация данных и structured output для LLM
- **python-dotenv** - для работы с переменными окружения
- **Make** - автоматизация сборки и запуска
- **ffmpeg** - конвертация голосовых (OGG/Opus → WAV 16 kHz mono)

Опционально для локального STT:
- **faster-whisper** - локальная транскрибация речи (CPU/GPU)

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
│   ├── bot.py          # Основной файл бота, инициализация aiogram
│   ├── handlers.py     # Обработчики команд и сообщений Telegram
│   ├── llm.py          # Работа с LLM через OpenRouter/Ollama
│   ├── models.py       # Pydantic модели для транзакций
│   ├── config.py       # Загрузка конфигурации из .env
│   └── stt.py          # Модуль STT (ffmpeg-конвертация и транскрибация)
├── prompts/
│   ├── system_prompt_text.txt   # Системный промпт для текстовых сообщений
│   └── system_prompt_image.txt  # Системный промпт для изображений
├── .env                # Переменные окружения (токены, настройки)
├── .env.example        # Пример конфигурации
├── pyproject.toml      # Конфигурация проекта для uv
├── Makefile            # Команды для запуска и управления
└── README.md           # Документация по запуску
```

**Принцип:** Всего 6 Python-файлов в одной папке `src/` (добавлен `stt.py` для голосовых). Никаких пакетов, подпакетов, сложной иерархии.

## Архитектура проекта

**Компоненты:**

1. **bot.py** - точка входа
   - Инициализирует aiogram Bot и Dispatcher
   - Регистрирует handlers
   - Запускает polling

2. **handlers.py** - обработка событий
   - `/start` - приветствие и очистка истории/транзакций
   - `/balance` - отчет о балансе и статистике
   - Обработчик текстовых сообщений → извлечение транзакций через LLM → сохранение транзакций → показ ответа + статус + баланс
   - Обработчик изображений → извлечение транзакций через VLM → сохранение транзакций → показ ответа + статус + баланс
   - Обработчик голосовых (`voice`/`audio`) → конвертация OGG/Opus → WAV (ffmpeg) → транскрибация (OpenAI Whisper API или faster-whisper) → извлечение транзакций как из текста
   - Хранит историю диалогов в памяти: `dict[int, list]` (chat_id → список сообщений)
   - Хранит транзакции в памяти: `dict[int, list[Transaction]]` (chat_id → список транзакций)

3. **llm.py** - интеграция с LLM
   - Метод `get_transaction_response_text()` - обработка текстовых сообщений со structured output
   - Метод `get_transaction_response_image()` - обработка изображений (VLM) со structured output
   - Единый интерфейс через AsyncOpenAI для OpenRouter и Ollama
   - Переключение между внешними и локальными моделями через конфигурацию

4. **models.py** - модели данных
   - Pydantic модели для транзакций (Transaction, TransactionResponse)
   - Enums для типов транзакций (TransactionType, TransactionFrequency)
   - Валидация данных транзакций

5. **config.py** - конфигурация
   - Класс Config с полями: `TELEGRAM_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_TEXT`, `MODEL_IMAGE`, `SYSTEM_PROMPT_TEXT`, `SYSTEM_PROMPT_IMAGE`
   - Загрузка промптов из файлов (`prompts/system_prompt_text.txt`, `prompts/system_prompt_image.txt`) или переменных окружения
   - Пути к файлам промптов можно задать через `SYSTEM_PROMPT_TEXT_PATH` и `SYSTEM_PROMPT_IMAGE_PATH` в .env
   - Загрузка из .env через python-dotenv
   - `MODEL_TEXT` - модель для текстовых сообщений, `MODEL_IMAGE` - модель для изображений (vision)

6. **stt.py** - распознавание речи (STT)
   - `convert_to_wav(input, output, 16kHz mono)` — конвертация через `ffmpeg`
   - `transcribe_audio(path_wav)` — вызов OpenAI Whisper API или локального `faster-whisper` на основе `STT_PROVIDER`
   - Конфигурация провайдеров через `.env`

**Поток данных (текстовые сообщения):**
```
Telegram → handlers.py (последнее сообщение) → llm.py (structured output) → OpenRouter/Ollama → 
llm.py → handlers.py (извлечь транзакции, сохранить в transactions, показать ответ + статус + баланс) → Telegram
```

**Поток данных (изображения):**
```
Telegram → handlers.py (изображение → base64) → llm.py (VLM + structured output) → OpenRouter/Ollama → 
llm.py → handlers.py (извлечь транзакции, сохранить в transactions, показать ответ + статус + баланс) → Telegram
```

**Принцип:** Никакой DI, никаких интерфейсов, никаких слоев абстракции. Просто прямые вызовы функций.

## Модель данных

**Хранение в памяти (без БД):**

Глобальные словари в `handlers.py`:
```python
chat_conversations: dict[int, list[dict]] = {}  # история диалогов (для контекста)
transactions: dict[int, list[Transaction]] = {}  # транзакции пользователей
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

**Структура транзакций:**
```python
from models import Transaction, TransactionType, TransactionFrequency

transactions[chat_id] = [
    Transaction(
        date=date(2024, 1, 15),
        time=time(14, 30),
        type=TransactionType.EXPENSE,
        amount=1500.0,
        frequency=TransactionFrequency.DAILY,
        category="продукты",
        description="Молоко, хлеб, яйца в магазине Пятёрочка"
    ),
    ...
]
```

**Pydantic модели (src/models.py):**

```python
from pydantic import BaseModel, Field
from datetime import date, time
from enum import Enum

class TransactionType(str, Enum):
    INCOME = "income"      # доход
    EXPENSE = "expense"    # расход

class TransactionFrequency(str, Enum):
    DAILY = "daily"           # повседневные
    PERIODIC = "periodic"     # периодические
    ONE_TIME = "one_time"     # разовые

class Transaction(BaseModel):
    date: date                           # дата транзакции
    time: time | None = None            # время (опционально)
    type: TransactionType                # доход/расход
    amount: float = Field(gt=0)          # сумма (строго положительная)
    frequency: TransactionFrequency       # тип (повседневные, периодические, разовые)
    category: str                        # категория (продукты, рестораны, такси и т.д.)
    description: str = ""                # описание транзакции (подробная информация о товарах, услугах, источнике, контрагенте и т.п.)

class TransactionResponse(BaseModel):
    transactions: list[Transaction] = []  # список транзакций (может быть пустым)
    answer: str                           # текстовый ответ пользователю (обязателен)
```

**Категории расходов/доходов:**
- Базовый список: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие
- LLM может предлагать новые категории, которые добавляются в список
- Если категория не определена - используется "другие"

**Операции:**
- При `/start` - очищаем историю и транзакции для данного чата
- При новом сообщении - извлекаем транзакции ТОЛЬКО из последнего сообщения
- Сохраняем транзакции в `transactions[chat_id]`
- При перезапуске бота - вся история и транзакции теряются

**Принцип:** Максимальная простота. Никаких БД, файлов, сериализации. Все данные живут только в runtime.

## Работа с LLM

**Используемая библиотека:** `openai` (официальный Python client, асинхронная версия)

**Настройка:**
```python
from openai import AsyncOpenAI

# Единый интерфейс для OpenRouter и Ollama
# Переключение моделей через изменение base_url и model в .env
client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,  # для Ollama можно использовать любое значение
    base_url=config.OPENAI_BASE_URL  # https://openrouter.ai/api/v1 или http://localhost:11434/v1
)
```

**Методы в llm.py:**

**1. Обработка текстовых сообщений:**
```python
async def get_transaction_response_text(
    last_message: str,
    message_history: list[dict]
) -> TransactionResponse:
    # Structured output через response_format с JSON schema из Pydantic
        response = await client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT_TEXT},
            *message_history[-10:],  # последние 10 сообщений для контекста
            {"role": "user", "content": last_message}
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "transaction_response",
            "schema": TransactionResponse.model_json_schema(),
            "strict": True
        }}
    )
    # Парсинг JSON ответа в TransactionResponse
    return TransactionResponse.model_validate_json(response.choices[0].message.content)
```

**2. Обработка изображений (VLM):**
```python
async def get_transaction_response_image(
    image_base64: str,
    message_history: list[dict]
) -> TransactionResponse:
    # Vision API с structured output
        response = await client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE},
            *message_history[-10:],
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Извлеки транзакции из этого изображения"}
                ]
            }
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "transaction_response",
            "schema": TransactionResponse.model_json_schema(),
            "strict": True
        }}
    )
    return TransactionResponse.model_validate_json(response.choices[0].message.content)
```

**Важные особенности:**
- **Structured output**: Использование Pydantic моделей для валидации ответов LLM через `response_format` с JSON schema
- **Извлечение транзакций**: ТОЛЬКО из последнего сообщения пользователя (не из всей истории) - подчеркнуто в системных промптах
- **Переключение моделей**: Единый интерфейс через AsyncOpenAI, переключение через изменение `OPENAI_BASE_URL` и `MODEL` в .env файле

**Параметры из .env:**
- `OPENAI_API_KEY` - ключ от OpenRouter (для Ollama можно любое значение)
- `OPENAI_BASE_URL` - URL API провайдера:
  - Для OpenRouter: `https://openrouter.ai/api/v1`
  - Для Ollama: `http://localhost:11434/v1`
- `MODEL_TEXT` - модель для обработки текстовых сообщений (например `openai/gpt-oss-20b:free` для OpenRouter или `llama3.2` для Ollama)
- `MODEL_IMAGE` - модель для обработки изображений, должна поддерживать vision (например `meta-llama/llama-3.2-11b-vision-instruct` для OpenRouter или `llama3.2-vision` для Ollama)
- `SYSTEM_PROMPT_TEXT_PATH` - путь к файлу с системным промптом для текстовых сообщений (по умолчанию: `prompts/system_prompt_text.txt`)
- `SYSTEM_PROMPT_IMAGE_PATH` - путь к файлу с системным промптом для изображений (по умолчанию: `prompts/system_prompt_image.txt`)
- `SYSTEM_PROMPT_TEXT` - альтернатива: системный промпт для текстовых сообщений напрямую (если указан, используется вместо файла)
- `SYSTEM_PROMPT_IMAGE` - альтернатива: системный промпт для изображений напрямую (если указан, используется вместо файла)

**Переключение между провайдерами:**
Для использования OpenRouter:
```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=meta-llama/llama-3.2-11b-vision-instruct
```

Для использования Ollama:
```bash
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_TEXT=llama3.2
MODEL_IMAGE=llama3.2-vision
```

**Обработка ошибок:**
- try/except для сетевых ошибок
- Возврат простого сообщения об ошибке пользователю
- Валидация ответов через Pydantic (автоматическая обработка ошибок парсинга)

**Принцип:** Асинхронный запрос-ответ с structured output. Никакого retry, никаких очередей, никакого streaming.

## Поток обработки голосовых

1. Получение `voice`/`audio` в Telegram (OGG/Opus)
2. Скачивание файла через Bot API
3. Конвертация `ffmpeg` → WAV 16 kHz mono (`pcm_s16le`)
4. Транскрибация:
   - OpenAI Whisper API (`STT_PROVIDER=openai`)
   - Локально: `faster-whisper` (`STT_PROVIDER=faster_whisper`)
5. Передача распознанного текста в `get_transaction_response_text()` для извлечения транзакций
6. Сохранение транзакций и расчёт баланса
7. Ответ пользователю: распознанный текст + результат извлечения + баланс

### Настройки для голосовых (STT)

```env
# Выбор провайдера: openai | faster_whisper
STT_PROVIDER=openai
STT_LANGUAGE=ru

# OpenAI Whisper API через OpenRouter
# (рекомендуемый вариант, т.к. остальной LLM тоже через OpenRouter)
OPENAI_STT_API_KEY=sk-or-...
OPENAI_AUDIO_BASE_URL=https://openrouter.ai/api/v1
STT_OPENAI_MODEL=openai/whisper-1

# Если аудио не поддерживается в вашем плане у OpenRouter,
# можно для STT указать прямой OpenAI endpoint:
# OPENAI_AUDIO_BASE_URL=https://api.openai.com/v1
# STT_OPENAI_MODEL=whisper-1

# Локально через faster-whisper
# STT_PROVIDER=faster_whisper
# STT_LOCAL_MODEL=small
# STT_DEVICE=cpu           # или cuda
# STT_COMPUTE_TYPE=int8    # или float16 (GPU)
```

Требуется установленный `ffmpeg`.

## Системные промпты

Промпты хранятся в файлах `prompts/system_prompt_text.txt` и `prompts/system_prompt_image.txt` для удобства чтения и редактирования. Пути к файлам можно настроить через переменные окружения `SYSTEM_PROMPT_TEXT_PATH` и `SYSTEM_PROMPT_IMAGE_PATH`.

**Для текстовых сообщений** (`prompts/system_prompt_text.txt`):
```
Ты персональный финансовый советник. Помогаешь пользователю вести учет доходов и расходов.

ВАЖНО: Транзакции извлекай ТОЛЬКО из последнего сообщения пользователя. Не используй информацию из предыдущих сообщений для создания транзакций.

НО: Для ответов на вопросы пользователя используй ВСЮ историю диалога! Если пользователь спрашивает про прошлые транзакции, события или информацию из истории - отвечай используя контекст предыдущих сообщений.

Твоя задача:
1. Извлечь финансовые транзакции из последнего сообщения пользователя (ТОЛЬКО из последнего!)
2. Для каждой транзакции создать объект со следующими полями (используй ТОЧНО эти названия полей на английском):
   - "date" (обязательно, формат YYYY-MM-DD строкой, используй сегодняшнюю дату если не указана)
   - "time" (если указано время, формат HH:MM:SS строкой, иначе null)
   - "type" (обязательно, строка: "income" для дохода или "expense" для расхода)
   - "amount" (обязательно, положительное число float, например 1500.0)
   - "frequency" (обязательно, строка: "daily" для повседневных, "periodic" для периодических, "one_time" для разовых)
   - "category" (обязательно, строка на русском: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие или новая категория по смыслу)
   - "description" (строка, подробная информация о товарах, услугах, источнике дохода, контрагенте и т.п., может быть пустой "")
3. Дать дружелюбный ответ пользователю в поле "answer":
   - Если пользователь задает вопрос - отвечай используя историю диалога
   - Если пользователь упоминает что-то из прошлого - используй информацию из истории для ответа
   - Если пользователь просто сообщает о транзакции - дай соответствующий комментарий

Категории: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие. Можешь предлагать новые категории, если они подходят лучше.

КРИТИЧЕСКИ ВАЖНО: Если в сообщении есть упоминание суммы денег (рублей, рублей, руб.) и что было куплено/продано/получено/потрачено - это ТРАНЗАКЦИЯ и её ОБЯЗАТЕЛЬНО нужно извлечь!

Пример правильного JSON ответа для сообщения "сегодня купил продукты на 1500 рублей":
{
  "transactions": [
    {
      "date": "2024-10-30",
      "time": null,
      "type": "expense",
      "amount": 1500.0,
      "frequency": "daily",
      "category": "продукты",
      "description": "Покупка продуктов на сумму 1500 рублей"
    }
  ],
  "answer": "Записал ваш расход на продукты в размере 1500 рублей."
}

Пример: Если пользователь ранее сообщил "Поел шаверму на 800р", а потом спрашивает "Чтото ел вкусное недавно. Что это было?" - ответь используя историю: "Вы недавно ели шаверму на 800 рублей."

Твой ответ должен быть в формате JSON с двумя полями:
- "transactions" - массив транзакций (обязательно, даже если пустой [])
- "answer" - текстовый ответ пользователю (обязательно), используй историю диалога для ответов на вопросы

Если в сообщении нет информации о транзакциях, верни пустой массив transactions [], но все равно дай полезный ответ пользователю используя историю диалога если это уместно.
```

**Для изображений** (`prompts/system_prompt_image.txt`):
```
Ты персональный финансовый советник. Помогаешь пользователю вести учет доходов и расходов.

ВАЖНО: Транзакции извлекай ТОЛЬКО из присланного изображения (чека, скриншота). Не используй информацию из предыдущих сообщений для создания транзакций.

Твоя задача:
1. Распознать текст на изображении (OCR)
2. Извлечь финансовые транзакции из изображения (чеки, выписки, скриншоты платежей)
3. Для каждой транзакции создать объект со следующими полями (используй ТОЧНО эти названия полей на английском):
   - "date" (обязательно, формат YYYY-MM-DD строкой)
   - "time" (если указано на изображении, формат HH:MM:SS строкой, иначе null)
   - "type" (обязательно, строка: "income" для дохода или "expense" для расхода)
   - "amount" (обязательно, положительное число float)
   - "frequency" (обязательно, строка: "daily" для повседневных, "periodic" для периодических, "one_time" для разовых)
   - "category" (обязательно, строка на русском: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие или новая категория по смыслу)
   - "description" (строка, подробная информация о товарах, услугах, источнике дохода, контрагенте и т.п., может быть пустой "")
4. Дать краткий комментарий о найденных транзакциях в поле "answer"

Категории: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие. Можешь предлагать новые категории, если они подходят лучше.

Твой ответ должен быть в формате JSON с двумя полями:
- "transactions" - массив транзакций (обязательно, даже если пустой [])
- "answer" - текстовый ответ пользователю (обязательно)

Если на изображении нет информации о транзакциях, верни пустой массив transactions [], но подтверди, что обработал изображение.
```

## Сценарии работы

**Сценарий 1: Первый запуск**
1. Пользователь отправляет `/start`
2. Бот отвечает приветственным сообщением
3. История диалога инициализируется с системным промптом
4. Список транзакций для пользователя очищается

**Сценарий 2: Обработка текстового сообщения с транзакцией**
1. Пользователь пишет: "Сегодня купил продукты на 1500 рублей"
2. Бот отправляет только последнее сообщение в LLM со structured output
3. LLM извлекает транзакцию и возвращает TransactionResponse
4. Бот сохраняет транзакцию в `transactions[chat_id]`
5. Бот рассчитывает баланс
6. Бот отправляет пользователю:
   - Ответ LLM из поля `answer`
   - Статус: "Найдено и сохранено 1 транзакция"
   - Текущий баланс: "Баланс: -1500 руб."

**Сценарий 3: Обработка изображения (чека)**
1. Пользователь отправляет изображение чека
2. Бот конвертирует изображение в base64
3. Бот отправляет изображение в VLM со structured output
4. VLM распознает текст и извлекает транзакции
5. Бот сохраняет транзакции в `transactions[chat_id]`
6. Бот рассчитывает баланс
7. Бот отправляет пользователю ответ + статус + баланс

**Сценарий 4: Запрос баланса**
1. Пользователь отправляет `/balance`
2. Бот рассчитывает из `transactions[chat_id]`:
   - Баланс = сумма доходов - сумма расходов
   - Общая сумма доходов
   - Общая сумма расходов
   - Статистика по категориям за все время
3. Бот отправляет форматированный отчет

**Сценарий 5: Сброс контекста**
1. Пользователь отправляет `/start`
2. История диалога и транзакции очищаются
3. Начинается новый диалог

**Ограничения:**
- Бот обрабатывает только текст и изображения (не PDF, не файлы других форматов)
- Один пользователь не блокирует других (асинхронность)
- При перезапуске бота все истории и транзакции теряются

## Подход к конфигурированию

**Файл .env** (не коммитится в git):
```bash
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=meta-llama/llama-3.2-11b-vision-instruct
SYSTEM_PROMPT_TEXT_PATH=prompts/system_prompt_text.txt
SYSTEM_PROMPT_IMAGE_PATH=prompts/system_prompt_image.txt
```

**Файл .env.example** (коммитится):
```bash
TELEGRAM_TOKEN=
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=meta-llama/llama-3.2-11b-vision-instruct
SYSTEM_PROMPT_TEXT_PATH=prompts/system_prompt_text.txt
SYSTEM_PROMPT_IMAGE_PATH=prompts/system_prompt_image.txt

# Альтернативно: можно переопределить промпты напрямую через переменные окружения
# SYSTEM_PROMPT_TEXT=
# SYSTEM_PROMPT_IMAGE=
```

**config.py:**
```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

def load_prompt(prompt_file_path: str, env_var: str = None) -> str:
    """Загружает промпт из файла или переменной окружения."""
    # Сначала пробуем загрузить из переменной окружения напрямую
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
    
    # Если переменной нет, пробуем загрузить из файла
    prompt_path = PROJECT_ROOT / prompt_file_path if not os.path.isabs(prompt_file_path) else Path(prompt_file_path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    
    return ""

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    MODEL_TEXT = os.getenv("MODEL_TEXT", os.getenv("MODEL"))  # Для обратной совместимости
    MODEL_IMAGE = os.getenv("MODEL_IMAGE")
    SYSTEM_PROMPT_TEXT = load_prompt(
        os.getenv("SYSTEM_PROMPT_TEXT_PATH", "prompts/system_prompt_text.txt"),
        "SYSTEM_PROMPT_TEXT"
    )
    SYSTEM_PROMPT_IMAGE = load_prompt(
        os.getenv("SYSTEM_PROMPT_IMAGE_PATH", "prompts/system_prompt_image.txt"),
        "SYSTEM_PROMPT_IMAGE"
    )

config = Config()
```

**Принцип загрузки промптов:**
1. Приоритет 1: Переменные окружения `SYSTEM_PROMPT_TEXT`/`SYSTEM_PROMPT_IMAGE` (если указаны напрямую)
2. Приоритет 2: Файлы по путям из `SYSTEM_PROMPT_TEXT_PATH`/`SYSTEM_PROMPT_IMAGE_PATH`
3. По умолчанию: `prompts/system_prompt_text.txt` и `prompts/system_prompt_image.txt`

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
- Ответы LLM (содержимое ответа + извлеченные транзакции)
- Ошибки при вызове LLM
- Исключения

**Что НЕ логируем:**
- Детальные трейсы успешных операций
- Метрики, аналитика

**Вывод:** Только в stdout/stderr (консоль)

**Принципы:**
- Без внешних библиотек (structlog и т.п.)
- Без файлов, ротации логов
- Без отправки в внешние системы
- Простой текстовый формат
