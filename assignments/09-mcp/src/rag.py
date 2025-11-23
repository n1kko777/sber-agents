import logging
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config import config

logger = logging.getLogger(__name__)

# Глобальные переменные
vector_store = None
retriever = None
chunks = None  # Для BM25 retriever
cross_encoder = None  # Для reranking (lazy loading)

def create_semantic_retriever():
    """Создание semantic retriever из vector store"""
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    return vector_store.as_retriever(
        search_kwargs={'k': config.SEMANTIC_RETRIEVER_K}
    )

def create_bm25_retriever():
    """Создание BM25 retriever из chunks"""
    if chunks is None or len(chunks) == 0:
        raise ValueError("Chunks not initialized for BM25")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.BM25_RETRIEVER_K
    return bm25

def create_hybrid_retriever():
    """Создание гибридного retriever (Semantic + BM25)"""
    semantic = create_semantic_retriever()
    bm25 = create_bm25_retriever()
    
    logger.info(f"Hybrid retriever: semantic_k={config.SEMANTIC_RETRIEVER_K}, bm25_k={config.BM25_RETRIEVER_K}")
    logger.info(f"Ensemble weights: semantic={config.ENSEMBLE_SEMANTIC_WEIGHT}, bm25={config.ENSEMBLE_BM25_WEIGHT}")
    
    return EnsembleRetriever(
        retrievers=[semantic, bm25],
        weights=[config.ENSEMBLE_SEMANTIC_WEIGHT, config.ENSEMBLE_BM25_WEIGHT]
    )

def get_cross_encoder():
    """Ленивая инициализация cross-encoder для reranking"""
    global cross_encoder
    if cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {config.CROSS_ENCODER_MODEL}")
            cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
            logger.info("✓ Cross-encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}", exc_info=True)
            raise
    return cross_encoder

def rerank_documents(query: str, documents: list, top_k: int = None):
    """
    Переранжирование документов с помощью cross-encoder
    
    Args:
        query: Запрос пользователя
        documents: Список Document объектов
        top_k: Количество документов для возврата (default: config.RERANKER_TOP_K)
    
    Returns:
        List[tuple]: Список (document, score) отсортированный по релевантности
    """
    if top_k is None:
        top_k = config.RERANKER_TOP_K
    
    if not documents:
        return []
    
    encoder = get_cross_encoder()
    
    # Создаем пары (query, document_text) для cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Cross-encoder оценивает релевантность каждой пары
    scores = encoder.predict(pairs)
    
    # Сортируем по убыванию score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
    
    # Возвращаем top_k наиболее релевантных
    return ranked[:top_k]

def create_retriever():
    """Фабрика для создания retriever по режиму"""
    mode = config.RETRIEVAL_MODE.lower()
    
    if mode == "semantic":
        logger.info("Creating semantic retriever")
        return create_semantic_retriever()
    
    elif mode == "hybrid":
        logger.info("Creating hybrid retriever (Semantic + BM25)")
        return create_hybrid_retriever()
    
    elif mode == "hybrid_reranker":
        logger.info("Creating hybrid retriever with reranker (Semantic + BM25 + Cross-encoder)")
        # Для hybrid_reranker используем тот же hybrid retriever
        # Reranking будет применен в get_rag_chain()
        return create_hybrid_retriever()
    
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}. Use 'semantic', 'hybrid', or 'hybrid_reranker'")

def initialize_retriever():
    """Инициализация retriever по режиму из конфига"""
    global retriever
    if vector_store is None:
        logger.error("Cannot initialize retriever: vector_store is None")
        return False
    
    try:
        retriever = create_retriever()
        logger.info(f"✓ Retriever initialized in '{config.RETRIEVAL_MODE}' mode")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}", exc_info=True)
        return False

def retrieve_documents(query: str):
    """
    Базовая функция поиска документов по запросу
    
    Args:
        query: Поисковый запрос
    
    Returns:
        list[Document]: Список найденных документов
    """
    if retriever is None:
        raise ValueError("Retriever not initialized")
    
    mode = config.RETRIEVAL_MODE.lower()
    
    # Для hybrid_reranker применяем reranking
    if mode == "hybrid_reranker":
        ensemble_docs = retriever.invoke(query)
        if not ensemble_docs:
            return []
        # Применяем reranking и возвращаем только документы
        reranked = rerank_documents(query, ensemble_docs, config.RERANKER_TOP_K)
        return [doc for doc, score in reranked]
    else:
        # Для semantic и hybrid - прямой вызов retriever
        return retriever.invoke(query)

def get_vector_store_stats():
    """Возвращает статистику векторного хранилища с полной информацией о конфигурации"""
    stats = {
        "status": "not initialized" if vector_store is None else "initialized",
        "count": 0,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "embedding_provider": config.EMBEDDING_PROVIDER,
    }
    
    if vector_store is not None:
        doc_count = len(vector_store.store) if hasattr(vector_store, 'store') else 0
        stats["count"] = doc_count
    
    # Добавляем информацию о моделях в зависимости от провайдера
    if config.EMBEDDING_PROVIDER == "openai":
        stats["embedding_model"] = config.EMBEDDING_MODEL
    elif config.EMBEDDING_PROVIDER == "huggingface":
        stats["embedding_model"] = config.HUGGINGFACE_EMBEDDING_MODEL
        stats["device"] = config.HUGGINGFACE_DEVICE
    
    # Добавляем параметры retrieval режима
    if config.RETRIEVAL_MODE == "semantic":
        stats["semantic_k"] = config.SEMANTIC_RETRIEVER_K
    elif config.RETRIEVAL_MODE == "hybrid":
        stats["semantic_k"] = config.SEMANTIC_RETRIEVER_K
        stats["bm25_k"] = config.BM25_RETRIEVER_K
        stats["semantic_weight"] = config.ENSEMBLE_SEMANTIC_WEIGHT
        stats["bm25_weight"] = config.ENSEMBLE_BM25_WEIGHT
    elif config.RETRIEVAL_MODE == "hybrid_reranker":
        stats["semantic_k"] = config.SEMANTIC_RETRIEVER_K
        stats["bm25_k"] = config.BM25_RETRIEVER_K
        stats["semantic_weight"] = config.ENSEMBLE_SEMANTIC_WEIGHT
        stats["bm25_weight"] = config.ENSEMBLE_BM25_WEIGHT
        stats["cross_encoder_model"] = config.CROSS_ENCODER_MODEL
        stats["reranker_top_k"] = config.RERANKER_TOP_K
    
    return stats

