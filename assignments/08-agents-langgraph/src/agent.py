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
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage

from config import config
from tools import rag_search, currency_converter, currency_rates, tavily_search

logger = logging.getLogger(__name__)


def create_bank_agent():
    """
    Создает ReAct агента для банковского ассистента используя create_agent() из LangChain 1.0
    
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
    
    # Инструменты которыми может пользоваться агент
    tools = [rag_search, currency_converter, currency_rates, tavily_search]
    
    # MemorySaver - сохраняет историю диалога в памяти (для многошагового диалога)
    # Каждый chat_id получает свою независимую историю
    checkpointer = MemorySaver()
    
    # create_agent() - API LangChain 1.0
    # Автоматически создает ReAct loop (цикл рассуждения и действий)
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    logger.info("✓ Bank agent created successfully using create_agent()")
    return agent_graph


# Глобальный экземпляр агента (создается один раз при старте бота)
bank_agent = None
_last_tool_graphs: dict[str, str | None] = {}


def initialize_agent():
    """
    Инициализация глобального экземпляра агента
    
    Паттерн singleton - создаем агента только один раз и переиспользуем
    """
    global bank_agent
    if bank_agent is None:
        bank_agent = create_bank_agent()
    return bank_agent


def get_last_tool_graph(chat_id: int) -> str | None:
    """Возвращает сохраненный граф вызовов инструментов для чата."""
    return _last_tool_graphs.get(str(chat_id))


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


def _shorten_text(text: str, max_length: int = 96) -> str:
    """Укорачивает текст для подписи на графе."""
    if not text:
        return ""
    clean = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3] + "..."


def _build_tool_call_graph(messages) -> str | None:
    """
    Собирает mermaid-граф вызовов инструментов за последний turn пользователя.
    """
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break
    if last_human_idx == -1:
        return None

    turn_messages = messages[last_human_idx:]
    user_text = _shorten_text(turn_messages[0].content if turn_messages else "")

    tool_calls: list[str] = []
    for msg in turn_messages:
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            name = getattr(call, "name", None)
            if name is None and isinstance(call, dict):
                name = call.get("name")
            if name:
                tool_calls.append(str(name))

    if not tool_calls:
        return None

    answer_text = ""
    if turn_messages:
        answer_text = _shorten_text(getattr(turn_messages[-1], "content", ""))

    lines = ["flowchart LR"]
    safe_user = user_text.replace('"', "'") or "Вопрос"
    lines.append(f'    user["User: {safe_user}"]')

    prev_node = "user"
    for idx, tool_name in enumerate(tool_calls, start=1):
        node_id = f"tool{idx}"
        safe_tool = tool_name.replace('"', "'")
        lines.append(f'    {node_id}["{idx}. {safe_tool}"]')
        lines.append(f"    {prev_node} --> {node_id}")
        prev_node = node_id

    if answer_text:
        safe_answer = answer_text.replace('"', "'")
        lines.append(f'    answer["Answer: {safe_answer}"]')
    else:
        lines.append('    answer["Answer"]')
    lines.append(f"    {prev_node} --> answer")

    return "\n".join(lines)


async def agent_answer(messages, chat_id: int):
    """
    Получить ответ от ReAct агента
    
    Процесс:
    1. Агент получает вопрос пользователя (HumanMessage)
    2. Рассуждает и решает нужен ли rag_search
    3. Если нужен - вызывает инструмент и получает контекст
    4. Формирует финальный ответ на основе контекста
    
    Используем stream для детального логирования каждого шага.
    История диалога сохраняется в MemorySaver по chat_id.
    
    Args:
        messages: Список LangChain messages (без SystemMessage, он уже в агенте)
        chat_id: ID чата для сохранения состояния диалога
    
    Returns:
        dict: {
            "answer": str - ответ агента пользователю,
            "documents": list - источники из rag_search (для SHOW_SOURCES и evaluation)
        }
    """
    if bank_agent is None:
        raise ValueError("Agent not initialized")
    
    inputs = {"messages": messages}
    # thread_id определяет отдельную историю диалога для каждого чата
    agent_config = {"configurable": {"thread_id": str(chat_id)}}
    
    logger.info(f"🤖 Agent starting for chat {chat_id}...")
    
    # stream() возвращает каждый шаг агента (для детального логирования)
    # stream_mode="values" - получаем полное состояние на каждом шаге
    final_state = None
    for state in bank_agent.stream(inputs, config=agent_config, stream_mode="values"):
        final_state = state
        _log_agent_step(state["messages"][-1])
    
    # Последнее сообщение - это финальный ответ агента
    last_message = final_state["messages"][-1]
    answer = last_message.content
    
    # Fallback для редких случаев когда LLM возвращает пустой ответ
    if not answer:
        logger.error(f"Empty answer from agent for chat {chat_id}")
        logger.debug(f"Last message type: {type(last_message).__name__}")
        logger.debug(f"Last message: {last_message}")
        answer = "Извините, не смог сформировать ответ. Попробуйте переформулировать вопрос."
    
    # Извлекаем documents только из текущего turn (для отображения источников)
    documents = _extract_documents_from_current_request(final_state["messages"])

    # Строим граф вызовов инструментов за текущий turn и сохраняем для /graph
    tool_graph = _build_tool_call_graph(final_state["messages"])
    _last_tool_graphs[str(chat_id)] = tool_graph
    
    logger.info(f"✅ Agent completed for chat {chat_id}")
    
    return {
        "answer": answer,
        "documents": documents,
        "tool_graph": tool_graph,
    }
