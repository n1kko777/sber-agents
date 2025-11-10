import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from config import config
import faq_lookup

logger = logging.getLogger(__name__)

# Глобальное векторное хранилище
vector_store = None
retriever = None

# Кеши для промптов и LLM клиентов
_conversational_answering_prompt = None
_retrieval_query_transform_prompt = None
_llm_query_transform = None
_llm = None

def initialize_retriever():
    """Инициализация retriever из векторного хранилища"""
    global retriever
    if vector_store is None:
        logger.error("Cannot initialize retriever: vector_store is None")
        return False
    
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K})
    logger.info(f"Retriever initialized with k={config.RETRIEVER_K}")
    return True

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
            temperature=0.4
        )
        logger.info(f"Query transform LLM initialized: {config.MODEL_QUERY_TRANSFORM}")
    return _llm_query_transform

def _get_llm():
    """Ленивая инициализация основной LLM с кешированием"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.MODEL,
            temperature=0.9
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

def _get_answer_chain():
    conversational_answering_prompt, _ = _load_prompts()
    return conversational_answering_prompt | _get_llm() | StrOutputParser()


def get_rag_chain():
    """Финальная RAG-цепочка с query transformation"""
    if retriever is None:
        raise ValueError("Retriever not initialized")
    
    conversational_answering_prompt, _ = _load_prompts()
    
    return (
        RunnablePassthrough.assign(
            context=get_retrieval_query_transformation_chain() | retriever | format_chunks
        )
        | conversational_answering_prompt
        | _get_llm()
        | StrOutputParser()
    )

async def rag_answer(messages):
    """
    Получить ответ от RAG с учетом истории диалога
    
    Args:
        messages: список LangChain messages (HumanMessage, AIMessage)
    
    Returns:
        str: ответ от RAG
    """
    if vector_store is None or retriever is None:
        logger.error("Vector store or retriever not initialized")
        raise ValueError("Векторное хранилище не инициализировано. Запустите индексацию.")
    
    last_user_message = messages[-1].content if messages else ""
    faq_document = faq_lookup.find_best_match(last_user_message)
    if faq_document:
        context = format_chunks([faq_document])
        answer_chain = _get_answer_chain()
        result = await answer_chain.ainvoke({"messages": messages, "context": context})
        return result
    
    rag_chain = get_rag_chain()
    result = await rag_chain.ainvoke({"messages": messages})
    return result

def get_vector_store_stats():
    """Возвращает статистику векторного хранилища"""
    if vector_store is None:
        return {"status": "not initialized", "count": 0}
    
    doc_count = len(vector_store.store) if hasattr(vector_store, 'store') else 0
    return {"status": "initialized", "count": doc_count}
