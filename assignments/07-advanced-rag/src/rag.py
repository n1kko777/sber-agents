import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config import config

logger = logging.getLogger(__name__)

# Глобальные переменные
vector_store = None
retriever = None
chunks = None  # Для BM25 retriever
cross_encoder = None  # Для reranking (lazy loading)

# Кеши для промптов и LLM клиентов
_conversational_answering_prompt = None
_retrieval_query_transform_prompt = None
_llm_query_transform = None
_llm = None

def get_last_user_query(messages) -> str:
    """Возвращает текст последнего пользовательского сообщения"""
    if not messages:
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    last_message = messages[-1]
    return getattr(last_message, "content", "")

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Удаляет дубликаты документов, сохраняя порядок появления"""
    unique_docs = []
    seen_keys = set()
    for doc in documents or []:
        page_content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("source", ""),
            metadata.get("page", ""),
            page_content.strip()
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_docs.append(doc)
    return unique_docs

def retrieve_documents_for_query(query: str) -> List[Document]:
    """Запрашивает документы у retriever с обработкой ошибок"""
    if retriever is None or not query:
        return []
    try:
        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(query)
        else:
            documents = retriever.get_relevant_documents(query)
        return documents or []
    except Exception as e:
        logger.error(f"Retrieval failed for query '{query[:50]}': {e}")
        return []

def collect_retrieval_documents(original_query: str, transformed_query: str) -> List[Document]:
    """Комбинирует результаты original + transformed запросов с дедупликацией"""
    documents = []
    documents.extend(retrieve_documents_for_query(original_query))
    if transformed_query and transformed_query != original_query:
        documents.extend(retrieve_documents_for_query(transformed_query))
    combined = deduplicate_documents(documents)
    logger.debug(
        "Combined %d unique documents from queries (original=%s, transformed=%s)",
        len(combined),
        bool(original_query),
        bool(transformed_query)
    )
    return combined

def rerank_with_cross_encoder(query: str, documents: List[Document], top_k: int = None) -> List[Document]:
    """Переранжирует документы и возвращает top_k Document"""
    if not documents:
        return []
    reranked = rerank_documents(
        query=query,
        documents=documents,
        top_k=top_k
    )
    if not reranked:
        return documents[:top_k] if top_k else documents
    return [doc for doc, _ in reranked]

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

def format_chunks(chunks):
    """
    Форматирование чанков с метаданными для лучшей прозрачности
    """
    if not chunks:
        return "Нет доступной информации"
    
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Получаем метаданные
        source = chunk.metadata.get('source', 'Unknown')
        page = chunk.metadata.get('page', 'N/A')
        
        # Извлекаем имя файла из пути
        source_name = source.split('/')[-1] if '/' in source else source
        
        # Форматируем чанк
        formatted_parts.append(
            f"[Источник {i}: {source_name}, стр. {page}]\n{chunk.page_content}"
        )
    
    return "\n\n---\n\n".join(formatted_parts)

def format_sources(documents):
    """
    Компактное форматирование источников с группировкой страниц по файлам
    Формат: "📚 Источники: file1.pdf (стр. 3, 5), file2.pdf (стр. 1)"
    """
    if not documents:
        return None
    
    # Группируем страницы по файлам
    sources_by_file = {}
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        source_name = source.split('/')[-1] if '/' in source else source
        page = doc.metadata.get('page', 'N/A')
        
        if source_name not in sources_by_file:
            sources_by_file[source_name] = []
        if page != 'N/A':
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

def _load_prompts():
    """Ленивая загрузка промптов с обработкой ошибок"""
    global _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    if _conversational_answering_prompt is not None:
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    try:
        conversation_system_text = config.load_prompt(config.CONVERSATION_SYSTEM_PROMPT_FILE)
        query_transform_text = config.load_prompt(config.QUERY_TRANSFORM_PROMPT_FILE)
        
        _conversational_answering_prompt = ChatPromptTemplate(
            [
                ("system", conversation_system_text),
                ("placeholder", "{messages}")
            ]
        )
        
        _retrieval_query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("user", query_transform_text),
            ]
        )
        
        logger.info("Prompts loaded successfully")
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
        
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompts: {e}", exc_info=True)
        raise

def _get_llm_query_transform():
    """Ленивая инициализация LLM для query transformation с кешированием"""
    global _llm_query_transform
    if _llm_query_transform is None:
        _llm_query_transform = ChatOpenAI(
            model=config.MODEL_QUERY_TRANSFORM,
            temperature=config.QUERY_TRANSFORM_TEMPERATURE
        )
        logger.info(f"Query transform LLM initialized: {config.MODEL_QUERY_TRANSFORM}")
    return _llm_query_transform

def _get_llm():
    """Ленивая инициализация основной LLM с кешированием"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.MODEL,
            temperature=config.LLM_TEMPERATURE
        )
        logger.info(f"Main LLM initialized: {config.MODEL}")
    return _llm

def get_retrieval_query_transformation_chain():
    """Цепочка трансформации запроса"""
    _, retrieval_query_transform_prompt = _load_prompts()
    return (
        retrieval_query_transform_prompt
        | _get_llm_query_transform()
        | StrOutputParser()
    )

def get_rag_chain():
    """Финальная RAG-цепочка возвращающая answer и documents в LCEL стиле"""
    if retriever is None:
        raise ValueError("Retriever not initialized")
    
    conversational_answering_prompt, _ = _load_prompts()
    mode = config.RETRIEVAL_MODE.lower()
    
    retrieval_chain = (
        RunnablePassthrough.assign(
            original_query=lambda x: get_last_user_query(x.get("messages", [])),
            transformed_query=get_retrieval_query_transformation_chain()
        )
        | RunnablePassthrough.assign(
            documents=lambda x: collect_retrieval_documents(
                x.get("original_query", ""),
                x.get("transformed_query", "")
            )
        )
    )
    
    if mode == "hybrid_reranker":
        retrieval_chain = retrieval_chain | RunnablePassthrough.assign(
            documents=lambda x: rerank_with_cross_encoder(
                query=x.get("original_query") or x.get("transformed_query") or "",
                documents=x["documents"],
                top_k=config.RERANKER_TOP_K
            )
        )
    
    return (
        retrieval_chain
        | RunnablePassthrough.assign(
            answer=lambda x: (conversational_answering_prompt | _get_llm() | StrOutputParser()).invoke({
                "context": format_chunks(x["documents"]),
                "messages": x["messages"]
            })
        )
        | (lambda x: {"answer": x["answer"], "documents": x["documents"]})
    )

async def rag_answer(messages):
    """
    Получить ответ от RAG с учетом истории диалога
    
    Args:
        messages: список LangChain messages (HumanMessage, AIMessage)
    
    Returns:
        dict: {"answer": str, "documents": list[Document]}
    """
    if vector_store is None or retriever is None:
        logger.error("Vector store or retriever not initialized")
        raise ValueError("Векторное хранилище не инициализировано. Запустите индексацию.")
    
    rag_chain = get_rag_chain()
    result = await rag_chain.ainvoke({"messages": messages})
    return result

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
