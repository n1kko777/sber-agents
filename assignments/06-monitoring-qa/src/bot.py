import asyncio
import logging
from pathlib import Path
from aiogram import Bot, Dispatcher
from handlers import router
from config import config
import indexer
import rag

# Создаем директорию для логов
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка логирования в консоль и файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler(log_dir / "bot.log", encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 50)
    logger.info("Bot starting...")
    
    # Индексация при старте
    logger.info("Starting indexing...")
    rag.vector_store = await indexer.reindex_all()
    if rag.vector_store:
        # Инициализируем retriever
        rag.initialize_retriever()
        stats = rag.get_vector_store_stats()
        logger.info(f"Indexing completed successfully: {stats['count']} documents indexed")
    else:
        logger.warning("Indexing completed with no documents - bot will run but cannot answer questions")
    
    bot = Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped with error: {e}", exc_info=True)
    finally:
        logger.info("Bot shutdown complete")
        logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

