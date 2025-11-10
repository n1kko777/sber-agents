import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import config
import indexer_with_json as indexer
import rag

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов в формате LangChain Messages
chat_conversations: dict[int, list] = {}

@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # Инициализируем историю с системным промптом в LangChain формате
    chat_conversations[message.chat.id] = [
        SystemMessage(content=config.SYSTEM_PROMPT)
    ]
    
    await message.answer(
        "Привет! Я RAG-ассистент Сбербанка.\n\n"
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
        "🤖 *RAG-ассистент Сбербанка*\n\n"
        "Я помогаю отвечать на вопросы по документам о кредитах и вкладах.\n\n"
        "*Доступные команды:*\n"
        "/start - Начать новый диалог (сбросить историю)\n"
        "/help - Показать эту справку\n"
        "/index - Переиндексировать документы\n"
        "/index\\_status - Проверить статус индексации\n\n"
        "*Возможности:*\n"
        "• Ответы на вопросы по документам\n"
        "• Понимание уточняющих вопросов\n"
        "• Сохранение контекста диалога\n\n"
        "*Примеры вопросов:*\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие проценты по вкладам?\n"
        "• Можно ли досрочно погасить кредит?\n\n"
        "_Если вопрос выходит за рамки документов, я сообщу об этом\\._"
    )
    await message.answer(help_text, parse_mode="Markdown")

@router.message(Command("index"))
async def cmd_index(message: Message):
    logger.info(f"User {message.chat.id} requested reindexing")
    await message.answer("Начинаю переиндексацию документов...")
    
    try:
        rag.vector_store = await indexer.reindex_all()
        if rag.vector_store:
            rag.initialize_retriever()
            stats = rag.get_vector_store_stats()
            await message.answer(
                f"✅ Переиндексация завершена!\n"
                f"Проиндексировано документов: {stats['count']}"
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
    else:
        await message.answer(
            f"📊 Статус индексации:\n"
            f"Статус: {stats['status']}\n"
            f"Количество документов: {stats['count']}"
        )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    # Инициализируем историю если её нет
    if message.chat.id not in chat_conversations:
        chat_conversations[message.chat.id] = [
            SystemMessage(content=config.SYSTEM_PROMPT)
        ]
    
    # Добавляем сообщение пользователя в историю
    chat_conversations[message.chat.id].append(
        HumanMessage(content=message.text)
    )
    
    try:
        # Проверка инициализации векторного хранилища
        if rag.vector_store is None or rag.retriever is None:
            logger.warning(f"Vector store not initialized for chat {message.chat.id}")
            await message.answer(
                "⚠️ Векторное хранилище не инициализировано. "
                "Пожалуйста, подождите или используйте /index для индексации."
            )
            # Удаляем последнее сообщение из истории
            chat_conversations[message.chat.id].pop()
            return
        
        # Получаем ответ через RAG (передаем историю без system message)
        response = await rag.rag_answer(chat_conversations[message.chat.id][1:])
        
        # Добавляем ответ в историю
        chat_conversations[message.chat.id].append(
            AIMessage(content=response)
        )
        
        await message.answer(response)
        
    except ValueError as e:
        logger.error(f"ValueError in handle_message for chat {message.chat.id}: {e}")
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await message.answer(
            "⚠️ Векторное хранилище не готово. "
            "Используйте /index для индексации документов."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for chat {message.chat.id}: {e}", exc_info=True)
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )
