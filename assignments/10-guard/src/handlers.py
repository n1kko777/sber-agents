import logging
from html import escape
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from langchain_core.messages import HumanMessage
from config import config
import indexer
import rag
import evaluation
import agent

logger = logging.getLogger(__name__)
router = Router()

# Глобальное хранилище для pending interrupts
# Ключ: chat_id, Значение: interrupt объект
pending_interrupts: dict[int, object] = {}


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
        "Я интеллектуальный агент\\, который *сам выбирает* какой инструмент использовать для ответа\\.\n\n"
        "*🧠 Как я работаю \\(ReAct\\):*\n"
        "1\\. *Думаю* \\(Reason\\) \\- анализирую ваш вопрос\n"
        "2\\. *Действую* \\(Act\\) \\- выбираю нужный инструмент\n"
        "3\\. *Отвечаю* \\- формирую ответ на основе полученных данных\n\n"
        "*🛠️ Мои инструменты:*\n"
        "📄 `rag_search` \\- поиск в PDF документах\n"
        "🏦 `search_products` \\- актуальные продукты банка \\(MCP\\)\n"
        "💱 `currency_converter` \\- курсы валют ЦБ РФ \\(MCP\\)\n"
        "💰 `deposit_income_calculator` \\- расчет дохода по вкладу \\(MCP\\)\n"
        "💳 `open_credit_card` \\- открытие карты \\(MCP\\, требует подтверждения\\)\n\n"
        "*📋 Доступные команды:*\n"
        "/start \\- Начать новый диалог\n"
        "/help \\- Показать эту справку\n"
        "/index \\- Переиндексировать документы\n"
        "/index\\_status \\- Статус и конфигурация\n"
        "/evaluate\\_dataset \\- Оценить качество RAG\n\n"
        "*💬 Примеры вопросов:*\n\n"
        "*Общие условия* \\(rag\\_search\\):\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие требования к заемщикам?\n"
        "• Можно ли досрочно погасить кредит?\n\n"
        "*Актуальные ставки* \\(search\\_products\\):\n"
        "• Какие сейчас ставки по вкладам?\n"
        "• Найди кредит до 500 тысяч\n"
        "• Какие кредитные карты есть?\n\n"
        "*Курсы валют* \\(currency\\_converter\\):\n"
        "• Какой курс доллара?\n"
        "• Сколько 1000 евро в рублях?\n\n"
        "*Расчет доходности* \\(deposit\\_income\\_calculator\\):\n"
        "• Посчитай доход с вклада 500 тысяч под 16% на год\n"
        "• Какой доход с капитализацией?\n"
        "• Рассчитай с учетом налогов\n\n"
        "*Открытие карты* \\(open\\_credit\\_card\\, требует подтверждения\\):\n"
        "• Открой мне кредитную карту\n"
        "• Хочу оформить дебетовую карту\n"
        "• Мне нужна новая карта\n\n"
        "_Используй /index\\_status для просмотра конфигурации\\._"
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
        
        # Проверяем на interrupt (требуется подтверждение пользователя)
        if result.get("interrupt"):
            interrupt_obj = result["interrupt"]
            
            # Сохраняем interrupt для последующей обработки
            pending_interrupts[message.chat.id] = interrupt_obj
            
            # Извлекаем детали операции
            action_request = interrupt_obj.value["action_requests"][0]
            tool_name = action_request["name"]
            tool_args = action_request["args"]
            
            # Форматируем сообщение для пользователя
            interrupt_message = (
                "⚠️ **Требуется подтверждение операции**\n\n"
                f"🔧 Инструмент: `{tool_name}`\n"
                f"📋 Параметры:\n"
            )
            
            for key, value in tool_args.items():
                interrupt_message += f"   • {key}: {value}\n"
            
            interrupt_message += "\n❓ Подтвердить выполнение операции?"
            
            # Создаем inline кнопки
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Подтвердить", 
                        callback_data=f"hitl_approve:{message.chat.id}"
                    ),
                    InlineKeyboardButton(
                        text="❌ Отклонить", 
                        callback_data=f"hitl_reject:{message.chat.id}"
                    )
                ]
            ])
            
            await message.answer(interrupt_message, reply_markup=keyboard, parse_mode="Markdown")
            logger.info(f"⏸️  Interrupt sent to user {message.chat.id}")
            return
        
        # Обычный ответ (без interrupt)
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


@router.callback_query(lambda c: c.data and c.data.startswith("hitl_"))
async def handle_hitl_callback(callback: CallbackQuery):
    """Обработка нажатий на кнопки HITL (Approve/Reject)"""
    try:
        # Парсим callback data
        action, chat_id_str = callback.data.split(":")
        chat_id = int(chat_id_str)
        
        # Проверяем что interrupt существует
        if chat_id not in pending_interrupts:
            await callback.answer("⚠️ Запрос устарел", show_alert=True)
            return
        
        # Удаляем кнопки
        await callback.message.edit_reply_markup(reply_markup=None)
        
        # Определяем решение
        if action == "hitl_approve":
            decision = "approve"
            await callback.message.edit_text(
                f"{escape(callback.message.text or '')}\n\n✅ <b>Операция подтверждена</b>",
                parse_mode="HTML"
            )
        else:  # hitl_reject
            decision = "reject"
            await callback.message.edit_text(
                f"{escape(callback.message.text or '')}\n\n❌ <b>Операция отклонена</b>",
                parse_mode="HTML"
            )
        
        # Удаляем из pending
        del pending_interrupts[chat_id]
        
        # Уведомляем пользователя о обработке
        processing_msg = await callback.message.answer("⏳ Обрабатываю решение...")
        
        # Резюмим агента
        result = await agent.agent_resume(
            chat_id=chat_id,
            decision=decision,
            message="Операция отклонена пользователем" if decision == "reject" else None
        )
        
        # Удаляем сообщение о обработке
        await processing_msg.delete()
        
        # Отправляем финальный ответ
        final_response = result["answer"]
        
        if config.SHOW_SOURCES and result["documents"]:
            sources = format_sources(result["documents"])
            if sources:
                final_response = f"{final_response}\n\n{sources}"
        
        await callback.message.answer(final_response)
        
        await callback.answer()
        logger.info(f"✓ HITL {decision} processed for chat {chat_id}")
        
    except Exception as e:
        logger.error(f"Error in handle_hitl_callback: {e}", exc_info=True)
        await callback.answer("❌ Ошибка обработки решения", show_alert=True)
