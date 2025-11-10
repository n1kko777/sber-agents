import logging
from difflib import SequenceMatcher
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_faq_documents: List[Document] = []


def update_faq_documents(documents: List[Document]) -> None:
    """Обновить локальный кеш FAQ-документов для быстрого поиска."""
    global _faq_documents
    _faq_documents = documents or []
    logger.info("FAQ lookup cache updated: %d entries", len(_faq_documents))


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def find_best_match(question: str, threshold: float = 0.82) -> Optional[Document]:
    """
    Находит наиболее похожий FAQ-документ по тексту вопроса.
    Возвращает None, если схожесть ниже порога.
    """
    if not _faq_documents or not question:
        return None
    
    normalized_query = _normalize(question)
    best_doc: Optional[Document] = None
    best_score = 0.0
    
    for doc in _faq_documents:
        normalized_question = doc.metadata.get("question_normalized")
        if not normalized_question:
            continue
        score = SequenceMatcher(None, normalized_query, normalized_question).ratio()
        if score > best_score:
            best_doc = doc
            best_score = score
    
    if best_doc and best_score >= threshold:
        logger.debug(
            "FAQ override hit: '%s' -> '%s' (score=%.2f)",
            question,
            best_doc.metadata.get("question"),
            best_score,
        )
        return best_doc
    
    return None
