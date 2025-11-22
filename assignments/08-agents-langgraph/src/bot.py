"""
Точка входа в Telegram бота с ReAct агентом

Последовательность запуска:
1. Настройка логирования
2. Индексация документов (PDF + JSON) в векторное хранилище
3. Инициализация RAG retriever (semantic/hybrid/hybrid_reranker)
4. Создание ReAct агента с MemorySaver
5. Запуск Telegram bot polling
"""
import os
import asyncio
import logging
from pathlib import Path

# Отключаем предупреждение tokenizers о параллелизме (для HuggingFace embeddings)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from aiogram import Bot, Dispatcher
from handlers import router
from config import config
import indexer
import rag
import agent

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
    logger.info("=" * 70)
    logger.info("🤖 ReAct Agent Bot Starting...")
    logger.info("=" * 70)
    
    # Логирование конфигурации
    logger.info("📋 Configuration:")
    logger.info(f"  Agent model: {config.MODEL}")
    logger.info(f"  Retrieval mode: {config.RETRIEVAL_MODE}")
    logger.info(f"  Embedding provider: {config.EMBEDDING_PROVIDER}")
    if config.EMBEDDING_PROVIDER == "openai":
        logger.info(f"  Embedding model: {config.EMBEDDING_MODEL}")
    elif config.EMBEDDING_PROVIDER == "huggingface":
        logger.info(f"  Embedding model: {config.HUGGINGFACE_EMBEDDING_MODEL}")
        logger.info(f"  Device: {config.HUGGINGFACE_DEVICE}")
    
    if config.RETRIEVAL_MODE in ["hybrid", "hybrid_reranker"]:
        logger.info(f"  Semantic k: {config.SEMANTIC_RETRIEVER_K}, BM25 k: {config.BM25_RETRIEVER_K}")
        logger.info(f"  Ensemble weights: {config.ENSEMBLE_SEMANTIC_WEIGHT}/{config.ENSEMBLE_BM25_WEIGHT}")
    if config.RETRIEVAL_MODE == "hybrid_reranker":
        logger.info(f"  Cross-encoder: {config.CROSS_ENCODER_MODEL}")
        logger.info(f"  Reranker top-k: {config.RERANKER_TOP_K}")
    
    logger.info(f"  LangSmith tracing: {config.LANGSMITH_TRACING_V2}")
    logger.info(f"  Show sources: {config.SHOW_SOURCES}")
    logger.info("-" * 70)
    
    # Индексация документов при старте
    # Загружаем PDF и JSON, создаем chunks, генерируем embeddings, сохраняем в FAISS
    logger.info("📚 Starting indexing...")
    result = await indexer.reindex_all()
    if result and result[0] is not None:
        rag.vector_store, rag.chunks = result
        # Инициализируем retriever (semantic/hybrid/hybrid_reranker в зависимости от конфига)
        rag.initialize_retriever()
        stats = rag.get_vector_store_stats()
        logger.info(f"✅ Indexing completed: {stats['count']} documents indexed")
    else:
        logger.warning("⚠️  Indexing completed with no documents - bot will run but cannot answer questions")
    
    # Инициализация ReAct агента
    # Создается один раз и переиспользуется для всех пользователей
    # MemorySaver внутри агента обеспечивает отдельную историю для каждого chat_id
    logger.info("🤖 Initializing ReAct agent...")
    agent.initialize_agent()
    logger.info("✅ Agent initialized successfully")
    
    bot = Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("-" * 70)
    logger.info("🚀 Starting bot polling...")
    logger.info("=" * 70)
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot stopped with error: {e}", exc_info=True)
    finally:
        logger.info("=" * 70)
        logger.info("🛑 Bot shutdown complete")
        logger.info("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())

