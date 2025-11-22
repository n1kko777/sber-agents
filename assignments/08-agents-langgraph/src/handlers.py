import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage
from config import config
import indexer
import rag
import evaluation
import agent

logger = logging.getLogger(__name__)
router = Router()


def format_sources(documents):
    """
    Компактное форматирование источников с группировкой страниц по файлам
    Формат: "📚 Источники: file1.pdf (стр. 3, 5), file2.pdf (стр. 1)"
    
    Args:
        documents: list[dict] с ключами "source" и опционально "page"
    """
    if not documents:
        return None
    
    # Группируем страницы по файлам
    sources_by_file = {}
    for doc in documents:
        source = doc.get('source', 'Unknown')
        source_name = source.split('/')[-1] if '/' in source else source
        page = doc.get('page')
        
        if source_name not in sources_by_file:
            sources_by_file[source_name] = []
        if page is not None:
            sources_by_file[source_name].append(str(page))
    
    # Форматируем компактно
    parts = []
    for filename, pages in sources_by_file.items():
        if pages:
            pages_str = ", ".join(sorted(set(pages), key=lambda x: int(x) if x.isdigit() else 0))
            parts.append(f"{filename} (стр. {pages_str})")
        else:
            parts.append(filename)
    
    return "📚 Источники: " + ", ".join(parts)


@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # История управляется агентом через MemorySaver (thread_id = chat_id)
    # Здесь только отправляем приветствие
    await message.answer(
        "Привет! Я ReAct Agent ассистент Сбербанка.\n\n"
        "Я могу:\n"
        "• Отвечать на вопросы по документам\n"
        "• Помогать с информацией о кредитах и вкладах\n"
        "• Поддерживать диалог с учетом контекста\n\n"
        "Используйте /help для просмотра всех команд."
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    logger.info(f"User {message.chat.id} requested help")
    help_text = (
        "🤖 *ReAct Agent ассистент Сбербанка*\n\n"
        "Я интеллектуальный агент\\, который *сам решает* когда нужно искать информацию в документах\\.\n\n"
        "*🧠 Как я работаю \\(ReAct\\):*\n"
        "1\\. *Думаю* \\(Reason\\) \\- анализирую ваш вопрос\n"
        "2\\. *Действую* \\(Act\\) \\- вызываю инструмент `rag_search` если нужна информация из документов\n"
        "3\\. *Отвечаю* \\- формирую ответ на основе найденных данных\n\n"
        "*📋 Доступные команды:*\n"
        "/start \\- Начать новый диалог\n"
        "/help \\- Показать эту справку\n"
        "/index \\- Переиндексировать документы\n"
        "/index\\_status \\- Статус и конфигурация\n"
        "/graph \\- Граф вызовов инструментов за последний запрос\n"
        "/evaluate\\_dataset \\- Оценить качество RAG\n\n"
        "*🔍 Режимы Retrieval:*\n"
        "• *semantic* \\- векторный поиск по смыслу\n"
        "• *hybrid* \\- Semantic \\+ BM25 \\(точные термины\\)\n"
        "• *hybrid\\_reranker* \\- Hybrid \\+ Cross\\-encoder\n\n"
        "*🧬 Embedding провайдеры:*\n"
        "• *openai* \\- облачные embeddings\n"
        "• *huggingface* \\- локальные модели\n\n"
        "*✨ Возможности:*\n"
        "• ReAct агент \\- самостоятельное принятие решений\n"
        "• История диалога через MemorySaver\n"
        "• Умный поиск в 3 режимах\n"
        "• RAGAS метрики качества\n"
        "• LangSmith трейсинг\n\n"
        "*💬 Примеры вопросов:*\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие проценты по вкладам?\n"
        "• Можно ли досрочно погасить кредит?\n"
        "• Привет \\(отвечу без поиска\\)\n\n"
        "_Используй /index\\_status для просмотра текущей конфигурации\\._"
    )
    await message.answer(help_text, parse_mode="MarkdownV2")

@router.message(Command("index"))
async def cmd_index(message: Message):
    logger.info(f"User {message.chat.id} requested reindexing")
    await message.answer("Начинаю переиндексацию документов...")
    
    try:
        result = await indexer.reindex_all()
        if result and result[0] is not None:
            rag.vector_store, rag.chunks = result
            rag.initialize_retriever()
            stats = rag.get_vector_store_stats()
            await message.answer(
                f"✅ Переиндексация завершена!\n"
                f"Проиндексировано документов: {stats['count']}\n"
                f"Режим: {stats['retrieval_mode']}\n"
                f"Провайдер: {stats['embedding_provider']}"
            )
        else:
            await message.answer("⚠️ Не найдено документов для индексации")
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        await message.answer(f"❌ Ошибка при переиндексации: {str(e)}")

@router.message(Command("index_status"))
async def cmd_index_status(message: Message):
    logger.info(f"User {message.chat.id} requested index status")
    stats = rag.get_vector_store_stats()
    
    if stats["status"] == "not initialized":
        await message.answer("⚠️ Векторное хранилище не инициализировано")
        return
    
    # Базовая информация
    status_text = (
        f"📊 *Статус индексации*\n"
            f"Статус: {stats['status']}\n"
        f"Документов: {stats['count']}\n\n"
        f"🔍 *Retrieval: {stats['retrieval_mode']}*\n"
    )
    
    # Параметры в зависимости от режима
    if stats['retrieval_mode'] == 'semantic':
        status_text += f"• Semantic k: {stats.get('semantic_k', 'N/A')}\n"
    elif stats['retrieval_mode'] == 'hybrid':
        status_text += (
            f"• Semantic k: {stats.get('semantic_k', 'N/A')}\n"
            f"• BM25 k: {stats.get('bm25_k', 'N/A')}\n"
            f"• Веса: {stats.get('semantic_weight', 0):.1f}/{stats.get('bm25_weight', 0):.1f}\n"
        )
    elif stats['retrieval_mode'] == 'hybrid_reranker':
        status_text += (
            f"• Semantic k: {stats.get('semantic_k', 'N/A')}\n"
            f"• BM25 k: {stats.get('bm25_k', 'N/A')}\n"
            f"• Reranker top k: {stats.get('reranker_top_k', 'N/A')}\n"
            f"• Cross-encoder: {stats.get('cross_encoder_model', 'N/A').split('/')[-1]}\n"
        )
    
    # Информация об embeddings
    status_text += f"\n🧬 *Embeddings: {stats['embedding_provider']}*\n"
    if stats['embedding_provider'] == 'openai':
        status_text += f"• Модель: {stats.get('embedding_model', 'N/A')}\n"
    elif stats['embedding_provider'] == 'huggingface':
        status_text += (
            f"• Модель: {stats.get('embedding_model', 'N/A').split('/')[-1]}\n"
            f"• Устройство: {stats.get('device', 'N/A')}\n"
        )
    
    await message.answer(status_text, parse_mode="Markdown")


@router.message(Command("graph"))
async def cmd_graph(message: Message):
    logger.info(f"User {message.chat.id} requested tool graph")
    graph_text = agent.get_last_tool_graph(message.chat.id)

    if not graph_text:
        await message.answer(
            "Пока нет графа: задайте вопрос, чтобы агент успел вызвать инструменты."
        )
        return

    response = (
        "Граф вызовов инструментов за последний запрос:\n"
        "```mermaid\n"
        f"{graph_text}\n"
        "```"
    )
    await message.answer(response, parse_mode="Markdown")


@router.message(Command("evaluate_dataset"))
async def cmd_evaluate_dataset(message: Message):
    logger.info(f"User {message.chat.id} requested dataset evaluation")
    
    # Проверка API ключа
    if not config.LANGSMITH_API_KEY:
        await message.answer(
            "⚠️ LangSmith API key не настроен.\n"
            "Установите LANGSMITH_API_KEY в .env файле для использования evaluation."
        )
        return
    
    # Проверка векторного хранилища
    if rag.vector_store is None or rag.retriever is None:
        await message.answer(
            "⚠️ Векторное хранилище не инициализировано.\n"
            "Используйте /index для индексации документов."
        )
        return
    
    # Извлекаем название датасета из команды (опционально)
    command_parts = message.text.split(maxsplit=1)
    dataset_name = command_parts[1] if len(command_parts) > 1 else None
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
        await message.answer(
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут...\n"
            f"Шаг 1/3: Запуск эксперимента в LangSmith..."
        )
    else:
        await message.answer(
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут..."
        )
    
    try:
        # Запускаем evaluation
        result = await evaluation.evaluate_dataset(dataset_name)
        
        # Формируем отчет
        metrics = result["metrics"]
        num_examples = result["num_examples"]
        
        report = (
            f"✅ Evaluation завершен!\n\n"
            f"📊 Датасет: {dataset_name}\n"
            f"📝 Примеров обработано: {num_examples}\n\n"
            f"🎯 RAGAS Метрики:\n"
        )
        
        # Добавляем метрики с описанием
        metric_descriptions = {
            "faithfulness": "Обоснованность (нет галлюцинаций)",
            "answer_relevancy": "Релевантность ответа",
            "answer_correctness": "Правильность ответа",
            "answer_similarity": "Похожесть на эталон",
            "context_recall": "Полнота контекста",
            "context_precision": "Точность поиска"
        }
        
        for metric_name, score in metrics.items():
            desc = metric_descriptions.get(metric_name, metric_name)
            # Эмодзи в зависимости от оценки
            if score >= 0.8:
                emoji = "🟢"
            elif score >= 0.6:
                emoji = "🟡"
            else:
                emoji = "🔴"
            report += f"{emoji} {desc}: {score:.3f}\n"
        
        report += "\n💡 Результаты загружены в LangSmith как feedback"
        
        await message.answer(report)
        logger.info(f"Evaluation completed for user {message.chat.id}")
        
    except ValueError as e:
        logger.error(f"ValueError in evaluation: {e}")
        await message.answer(f"❌ Ошибка: {str(e)}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при evaluation:\n{str(e)}\n\n"
            f"Проверьте логи для подробностей."
        )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    try:
        # Проверка инициализации векторного хранилища
        if rag.vector_store is None or rag.retriever is None:
            logger.warning(f"Vector store not initialized for chat {message.chat.id}")
            await message.answer(
                "⚠️ Векторное хранилище не инициализировано. "
                "Пожалуйста, подождите или используйте /index для индексации."
            )
            return
        
        # Создаем сообщение в формате LangChain
        user_message = HumanMessage(content=message.text)
        
        # Получаем ответ через ReAct агента
        # ВАЖНО: Передаем только текущее сообщение, а не всю историю!
        # История хранится в агенте (MemorySaver) и управляется через chat_id
        # Агент сам решает:
        # - Нужно ли использовать rag_search
        # - Сколько раз его вызвать
        # - Как сформировать ответ на основе контекста
        result = await agent.agent_answer(
            [user_message],
            message.chat.id
        )
        
        # Формируем итоговый ответ для пользователя
        final_response = result["answer"]
        
        # Опционально добавляем источники (если SHOW_SOURCES=true)
        # documents содержат только источники из текущего ответа, не из всей истории
        if config.SHOW_SOURCES and result["documents"]:
            sources = format_sources(result["documents"])
            if sources:
                final_response = f"{final_response}\n\n{sources}"
        
        await message.answer(final_response)
        
    except ValueError as e:
        logger.error(f"ValueError in handle_message for chat {message.chat.id}: {e}")
        await message.answer(
            "⚠️ Векторное хранилище не готово. "
            "Используйте /index для индексации документов."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for chat {message.chat.id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )
