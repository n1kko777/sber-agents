"""
ReAct агент для банковского ассистента

ReAct = Reasoning + Acting - паттерн где агент:
1. Рассуждает (Reasoning) - анализирует вопрос и решает что делать
2. Действует (Acting) - вызывает инструменты (tools) для получения информации
3. Повторяет цикл до получения ответа

Используем упрощенный подход create_agent() из LangChain 1.0 вместо ручного LangGraph.
"""
import json
import logging

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import PIIMiddleware

from config import config
from tools import rag_search

logger = logging.getLogger(__name__)


async def create_bank_agent():
    """
    Создает ReAct агента для банковского ассистента используя create_agent() из LangChain 1.0
    
    Подключает три типа инструментов:
    1. rag_search - поиск в статических PDF документах
    2. search_products - поиск актуальных продуктов банка (MCP)
    3. currency_converter - конвертация валют (MCP)
    
    Returns:
        Скомпилированный агент LangChain 1.0 с MemorySaver для сохранения истории диалогов
    """
    logger.info("Creating bank agent using create_agent()...")
    
    # Загружаем системный промпт из файла (удобнее редактировать отдельно)
    system_prompt = config.load_prompt(config.AGENT_SYSTEM_PROMPT_FILE)
    
    # Инициализируем LLM (модель которая будет рассуждать и принимать решения)
    llm = ChatOpenAI(
        model=config.MODEL,
        temperature=0.7  # Умеренная креативность для естественных ответов
    )
    
    # Базовый инструмент - поиск в PDF документах
    tools = [rag_search]
    
    # Подключаем MCP инструменты (search_products, currency_converter)
    if config.MCP_ENABLED:
        try:
            logger.info(f"Connecting to MCP server '{config.MCP_SERVER_NAME}' at {config.MCP_SERVER_URL}...")
            
            # Создаем MCP клиент для подключения к MCP серверу
            mcp_client = MultiServerMCPClient({
                config.MCP_SERVER_NAME: {
                    "transport": config.MCP_SERVER_TRANSPORT,
                    "url": config.MCP_SERVER_URL
                }
            })
            
            # Получаем инструменты от MCP сервера
            mcp_tools = await mcp_client.get_tools()
            
            if mcp_tools:
                tools.extend(mcp_tools)
                logger.info(f"✓ Connected to MCP server, loaded {len(mcp_tools)} tools:")
                for tool in mcp_tools:
                    logger.info(f"  - {tool.name}: {tool.description}")
            else:
                logger.warning("⚠️  MCP server connected but no tools returned")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to connect to MCP server: {e}")
            logger.warning("   Agent will work without MCP tools (search_products, currency_converter)")
            logger.warning("   To enable MCP tools, start the server: make run-mcp-bank")
    else:
        logger.info("ℹ️  MCP is disabled (MCP_ENABLED=false), agent will use only rag_search")
    
    # MemorySaver - сохраняет историю диалога в памяти (для многошагового диалога)
    # Каждый chat_id получает свою независимую историю
    checkpointer = MemorySaver()
    
    # create_agent() - API LangChain 1.0
    # Автоматически создает ReAct loop (цикл рассуждения и действий)
    # С Human-in-the-Loop middleware для критичных операций
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            # 🔒 Layer 1: Model Call Limit
            # Максимум 2 вызова модели за один запуск
            ModelCallLimitMiddleware(run_limit=20),
            
            # 🔒 Layer 2: Tool Call Limit
            # Максимум 2 вызова инструментов за один запуск
            ToolCallLimitMiddleware(run_limit=20),
            
            # 🔒 Layer 3: PII Protection
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=False,
                apply_to_output=True
            ),
            
            # 🔒 Layer 4: Human-in-the-Loop
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "open_credit_card": {
                        "allowed_decisions": ["approve", "reject"]
                    },
                    "open_deposit": {
                        "allowed_decisions": ["approve", "reject"]
                    }
                }
            )
        ]
    )

    logger.info(f"✓ Bank agent created successfully with {len(tools)} tools and HITL middleware")
    return agent_graph


# Глобальный экземпляр агента (создается один раз при старте бота)
bank_agent = None


async def initialize_agent():
    """
    Инициализация глобального экземпляра агента
    
    Паттерн singleton - создаем агента только один раз и переиспользуем
    Асинхронная функция так как подключение к MCP серверу асинхронное
    """
    global bank_agent
    if bank_agent is None:
        bank_agent = await create_bank_agent()
    return bank_agent


def _log_agent_step(msg):
    """
    Логирует один шаг работы агента для отладки
    
    Помогает понять что происходит внутри агента на каждом шаге ReAct цикла:
    - HumanMessage: вопрос пользователя
    - AIMessage с tool_calls: агент решил вызвать инструмент
    - ToolMessage: результат выполнения инструмента
    - AIMessage с content: финальный ответ агента
    
    Args:
        msg: сообщение из stream
    """
    msg_type = type(msg).__name__
    logger.info(f"  Step: {msg_type}")
    
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        # AIMessage с вызовом инструмента - агент решил что нужна доп. информация
        for tc in msg.tool_calls:
            logger.info(f"    🔧 Tool: {tc['name']}")
            logger.info(f"    Args: {tc['args']}")
    elif hasattr(msg, 'name') and msg.name:
        # ToolMessage - результат работы инструмента
        logger.info(f"    📦 Tool: {msg.name}")
        logger.info(f"    Result: {str(msg.content)[:200]}...")
    elif hasattr(msg, 'content'):
        # Обычное сообщение (вопрос пользователя или финальный ответ)
        content_preview = str(msg.content)[:100] if msg.content else ""
        if content_preview:
            logger.info(f"    Content: {content_preview}...")
        else:
            # Пустой content в AIMessage - редкий глюк LLM
            if msg_type == "AIMessage":
                logger.warning("    ⚠️ AIMessage with empty content and no tool_calls!")


def _extract_documents_from_current_request(messages):
    """
    Извлекает documents из всех ToolMessage с rag_search после последнего HumanMessage
    
    ВАЖНО: Берем только текущий turn (после последнего вопроса пользователя),
    НЕ всю историю диалога! Это нужно для:
    1. Показа источников только для текущего ответа (SHOW_SOURCES)
    2. Правильной оценки контекста в RAGAS evaluation
    
    Агент может вызвать rag_search несколько раз за один turn - собираем все.
    
    Args:
        messages: список сообщений из final_state агента
    
    Returns:
        list[dict]: список documents с ключами "source", "page_content" и опционально "page"
    """
    documents = []
    
    # Находим индекс последнего HumanMessage (начало текущего turn)
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break
    
    # Собираем все ToolMessage с rag_search после последнего HumanMessage
    if last_human_idx != -1:
        for msg in messages[last_human_idx:]:
            if isinstance(msg, ToolMessage) and msg.name == "rag_search":
                try:
                    data = json.loads(msg.content)
                    sources = data.get("sources", [])
                    documents.extend(sources)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse rag_search result as JSON")
    
    return documents


async def _run_agent_stream(inputs, agent_config, chat_id: int):
    """
    Общая функция для обработки agent stream (для agent_answer и agent_resume)
    
    Обрабатывает stream от агента, проверяет на interrupts и возвращает результат.
    Используется как agent_answer, так и agent_resume для избежания дублирования кода.
    
    Args:
        inputs: dict с messages или Command объект для resume
        agent_config: конфигурация агента с thread_id
        chat_id: ID чата для логирования
    
    Returns:
        dict: {
            "answer": str | None - ответ агента (None если interrupt),
            "documents": list - источники из rag_search,
            "interrupt": object | None - interrupt объект если требуется подтверждение
        }
    """
    if bank_agent is None:
        raise ValueError("Agent not initialized")
    
    interrupts = []
    final_state = None
    
    # Обработка stream с проверкой на interrupts
    # astream() возвращает каждый шаг агента асинхронно
    # ВАЖНО: используем astream() т.к. MCP инструменты асинхронные
    async for step in bank_agent.astream(inputs, config=agent_config):
        # Проверяем на interrupt через специальный __interrupt__ ключ
        if "__interrupt__" in step:
            interrupt_data = step["__interrupt__"]
            if isinstance(interrupt_data, tuple) and interrupt_data:
                interrupts.append(interrupt_data[0])
                logger.info(f"⚠️  INTERRUPT detected: {interrupt_data[0].id}")
        
        # Обычное обновление состояния (логируем шаги)
        for node_name, update in step.items():
            if node_name != "__interrupt__" and isinstance(update, dict) and "messages" in update:
                final_state = update
                _log_agent_step(update["messages"][-1])
    
    # Если есть interrupt - возвращаем его (агент остановлен)
    if interrupts:
        logger.info(f"🛑 Agent stopped with interrupt for chat {chat_id}")
        return {
            "answer": None,
            "documents": [],
            "interrupt": interrupts[0]
        }
    
    # Получаем полное состояние агента после завершения
    # ВАЖНО: final_state из stream содержит только последнее обновление,
    # но нам нужны ВСЕ сообщения для извлечения documents
    full_state = bank_agent.get_state(agent_config)
    all_messages = full_state.values["messages"]
    
    # Обычный ответ (без interrupt)
    last_message = all_messages[-1]
    answer = last_message.content
    
    # Fallback для редких случаев когда LLM возвращает пустой ответ
    if not answer:
        logger.error(f"Empty answer from agent for chat {chat_id}")
        logger.debug(f"Last message type: {type(last_message).__name__}")
        logger.debug(f"Last message: {last_message}")
        answer = "Извините, не смог сформировать ответ. Попробуйте переформулировать вопрос."
    
    # Извлекаем documents только из текущего turn (для отображения источников)
    logger.info(f"Extracting documents from full state with {len(all_messages)} messages")
    documents = _extract_documents_from_current_request(all_messages)
    
    logger.info(f"✅ Agent completed for chat {chat_id}")
    logger.info(f"📚 Documents extracted: {len(documents)} documents")
    
    return {
        "answer": answer,
        "documents": documents,
        "interrupt": None
    }


async def agent_answer(messages, chat_id: int):
    """
    Получить ответ от ReAct агента с поддержкой Human-in-the-Loop
    
    Процесс:
    1. Агент получает вопрос пользователя (HumanMessage)
    2. Рассуждает и решает нужен ли инструмент (rag_search, search_products и т.д.)
    3. Если нужен - вызывает инструмент и получает данные
    4. Если инструмент критичный (open_credit_card) - создается interrupt
    5. Формирует финальный ответ на основе контекста
    
    История диалога сохраняется в MemorySaver по chat_id.
    
    Args:
        messages: Список LangChain messages (без SystemMessage, он уже в агенте)
        chat_id: ID чата для сохранения состояния диалога
    
    Returns:
        dict: {
            "answer": str | None - ответ агента (None если interrupt),
            "documents": list - источники из rag_search (для SHOW_SOURCES и evaluation),
            "interrupt": object | None - interrupt объект если требуется подтверждение
        }
    """
    inputs = {"messages": messages}
    # thread_id определяет отдельную историю диалога для каждого чата
    agent_config = {"configurable": {"thread_id": str(chat_id)}}
    
    logger.info(f"🤖 Agent starting for chat {chat_id}...")
    
    return await _run_agent_stream(inputs, agent_config, chat_id)


async def agent_resume(chat_id: int, decision: str, message: str = None):
    """
    Возобновить выполнение агента после Human-in-the-Loop interrupt
    
    После того как пользователь принял решение (approve/reject) по критичной операции,
    эта функция продолжает выполнение агента с учетом решения.
    
    Args:
        chat_id: ID чата для восстановления контекста диалога
        decision: "approve" или "reject" - решение пользователя
        message: Сообщение при reject (причина отклонения), опционально
    
    Returns:
        dict: аналогично agent_answer - {answer, documents, interrupt}
    """
    from langgraph.types import Command
    
    # thread_id для восстановления контекста диалога
    agent_config = {"configurable": {"thread_id": str(chat_id)}}
    
    logger.info(f"🔄 Resuming agent for chat {chat_id} with decision: {decision}")
    
    # Формируем команду resume согласно API LangChain
    if decision == "approve":
        command = Command(resume={"decisions": [{"type": "approve"}]})
    else:  # reject
        command = Command(resume={
            "decisions": [{
                "type": "reject",
                "message": message or "Операция отклонена пользователем"
            }]
        })
    
    # Продолжаем выполнение агента с решением пользователя
    return await _run_agent_stream(command, agent_config, chat_id)
