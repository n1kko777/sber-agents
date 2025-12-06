# 🎫 Local Stdio MCP Ticket Server (`mcp-local-stdio`)

Демонстрационный проект, показывающий интеграцию **Model Context Protocol (MCP)** через stdio транспорт с системами поддержки клиентов и LangChain для создания AI-агентов.

## 📋 Описание проекта

Этот проект состоит из двух основных компонентов:

### 🚀 MCP Ticket Server (`server/`)
Сервер реализует протокол MCP и предоставляет инструменты для поиска тикетов в базе данных поддержки:

- **База данных**: Excel файл с тикетами поддержки (`server/data/requests.xls`)
- **Инструмент**: `search_tickets` - поиск тикетов по различным критериям
- **Автогенерация данных**: При первом запуске создает 50 образцов тикетов

**Поддерживаемые критерии поиска:**
- `user_id` - ID пользователя
- `status` - статус тикета (open, closed, pending, in_progress)
- `priority` - приоритет (low, medium, high, critical)  
- `category` - категория (authentication, billing, feature, technical, security)
- `keyword` - поиск по заголовку и описанию

### 🤖 LangChain Client (`client/`)
Клиент демонстрирует интеграцию MCP сервера с LangChain для создания AI-агента поддержки:

- **AI Модель**: OpenAI GPT-4o-mini
- **Фреймворк**: LangGraph с ReAct агентом
- **Интерфейс**: Rich консольный интерфейс
- **Режимы**: Демо и интерактивный

## ⚡ Быстрый старт

Хотите сразу попробовать? Выполните 3 команды:

```bash
# 1. Установите зависимости
uv sync

# 2. Запустите MCP Inspector (откроется в браузере)
uv run mcp dev server/main.py:mcp

# 3. В Inspector нажмите "Connect" и тестируйте инструмент search_tickets
```

MCP Inspector позволяет тестировать сервер **без необходимости API ключей** и LLM!

## 🛠️ Установка и настройка

### Требования
- Python 3.12+
- OpenAI API ключ (для клиента)
- uv (рекомендуется для управления зависимостями)

### Установка зависимостей

```bash
# Перейдите в директорию
cd mcp/mcp-local-stdio

# Установите зависимости через uv
uv sync

# Или через pip
pip install -e .
```

### Настройка OpenAI API (для клиента)

```bash
# Создайте .env файл с вашим API ключом
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## 🚀 Запуск

### Способ 1: Тестирование через MCP Inspector (рекомендуется)

MCP Inspector - официальный инструмент для отладки и тестирования MCP серверов:

```bash
# Запуск MCP Inspector (автоматически откроет браузер)
uv run mcp dev server/main.py:mcp

# После запуска:
# - MCP Inspector откроется в браузере на http://localhost:6274
# - Proxy сервер будет доступен на localhost:6277
# - Можно тестировать инструменты без необходимости LLM
```

### Способ 2: Запуск LangChain клиента с AI агентом

Для работы требуется OpenAI API ключ:

```bash
# 1. Создайте .env файл с вашим API ключом
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# 2. Запустите клиента
uv run client/simple.py

# Выберите режим:
# - demo: демонстрационные запросы
# - interactive: интерактивный режим для своих вопросов
```

### Способ 3: Генерация тестовых данных

```bash
# Создание образцов данных (если нужно пересоздать базу)
uv run server/main.py
```

### Способ 4: Интеграция с MCP клиентами (Claude Desktop, Cursor, и др.)

Сервер совместим с различными MCP клиентами. Пример конфигурации для Claude Desktop:

```json
{
  "mcpServers": {
    "ticket-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/your-username/path/to/mcp-local-stdio",
        "run",
        "--with", "mcp",
        "--with", "openpyxl",
        "--with", "pandas",
        "mcp", "run", 
        "server/main.py:mcp"
      ]
    }
  }
}
```

**Важно:** Замените `/Users/your-username/path/to/mcp-server-demo` на полный путь к проекту.

## 📝 Примеры использования

### Тестирование через MCP Inspector

После запуска `uv run mcp dev server/main.py:mcp` откройте Inspector в браузере и попробуйте:

**1. Поиск критических тикетов по безопасности:**
```json
{
  "status": "open",
  "priority": "critical",
  "category": "security"
}
```

**2. Поиск всех тикетов пользователя:**
```json
{
  "user_id": "user123"
}
```

**3. Поиск по ключевым словам:**
```json
{
  "keyword": "login",
  "category": "authentication"
}
```

**4. Комбинированный поиск:**
```json
{
  "status": "open",
  "priority": "high",
  "keyword": "payment"
}
```

### Программное использование

```python
# Через MCP инструмент в коде
search_tickets(status="open", priority="high")
search_tickets(user_id="user123")
search_tickets(keyword="login", category="authentication")
```

## 🔧 Структура проекта

```
mcp-local-stdio/
├── server/
│   ├── main.py           # MCP сервер с инструментом search_tickets
│   ├── sample_data.py    # Генератор образцов данных
│   └── data/
│       └── requests.xls  # База данных тикетов (создается автоматически)
├── client/
│   └── simple.py         # LangChain клиент с демо-интерфейсом
├── pyproject.toml        # Конфигурация проекта и зависимости
└── README.md            # Документация
```

## 🤖 Интеграция с LangChain

Проект демонстрирует использование [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) для создания AI-агентов:

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# Подключение к MCP серверу
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Загрузка MCP инструментов как LangChain Tools
        tools = await load_mcp_tools(session)
        
        # Создание ReAct агента с MCP инструментами
        agent = create_react_agent(model, tools)
```

## 📚 Полезные ссылки

### Документация MCP
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk) - Основная документация Python SDK
- [MCP Specification](https://spec.modelcontextprotocol.io/) - Спецификация протокола
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) - Официальный инструмент для отладки и тестирования MCP серверов

### Интеграция с LangChain
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters) - Адаптеры для интеграции MCP с LangChain
- [LangGraph MCP Reference](https://langchain-ai.github.io/langgraph/reference/mcp/) - Документация по использованию MCP в LangGraph
- [LangGraph MCP Tutorial](https://langchain-ai.github.io/langgraph/concepts/mcp/) - Документация по использованию MCP в LangGraph

### Дополнительные ресурсы
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers) - Коллекция готовых MCP серверов
- [Claude Desktop MCP Setup](https://claude.ai/docs/mcp) - Настройка MCP в Claude Desktop
- [FastMCP Tutorial](https://www.firecrawl.dev/blog/fastmcp-tutorial-building-mcp-servers-python) - Гайд по созданию MCP сервера на Python
- [Introduction to Model Context Protocol](https://anthropic.skilljar.com/introduction-to-model-context-protocol) - Курс "Введение в MCP" от Anthropic
- [FastMCP](https://gofastmcp.com/servers/server) - Гайд по созданию MCP сервера на Python