import logging
from pathlib import Path
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from config import config
import faq_lookup

logger = logging.getLogger(__name__)

def _normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_json_documents(json_file_path: str) -> list:
    """
    Загрузка документов из JSON файла с вопросами-ответами
    Каждая пара Q&A становится отдельным чанком
    """
    from pathlib import Path
    
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    try:
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON {json_file_path}: {exc}")
        return []
    
    documents: list[Document] = []
    seen_pairs = set()
    
    for item in raw_data:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        
        qa_key = (question, answer)
        if qa_key in seen_pairs:
            continue
        seen_pairs.add(qa_key)
        
        metadata = {
            "source": str(json_path),
            "category": item.get("category"),
            "url": item.get("url"),
            "type": item.get("type", "faq"),
            "question": question,
            "question_normalized": _normalize_question(question),
        }
        page_content = f"Вопрос: {question}\nОтвет: {answer}"
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    logger.info(f"Loaded {len(documents)} unique Q&A pairs from JSON")
    return documents

def load_pdf_documents(data_dir: str) -> list:
    """Загрузка всех PDF документов из директории"""
    pages = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory {data_dir} does not exist")
        return pages
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages.extend(loader.load())
        logger.info(f"Loaded {pdf_file.name}")
    
    return pages

def split_documents(pages: list) -> list:
    """Разбиение документов с учетом структуры"""
    if not pages:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=120,
        separators=[
            "\n\n",       # основной разделитель параграфов
            "\n•",        # маркеры списков
            "\n- ",       # маркеры списков
            "\n— ",       # списки с тире
            "\n",         # одиночные переносы
            ". ",         # конец предложения
            " ",          # слова
            ""            # fallback до символов
        ],
        keep_separator=True,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks (chunk_size=450, overlap=120)")
    return chunks

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)"""
    logger.info("Starting full reindexing...")
    
    try:
        # 1. Загружаем и обрабатываем PDF документы
        pdf_pages = load_pdf_documents(config.DATA_DIR)
        if not pdf_pages:
            logger.warning("No PDF documents found to index")
        
        pdf_chunks = split_documents(pdf_pages) if pdf_pages else []
        
        # 2. Загружаем JSON с вопросами-ответами
        json_chunks = []
        json_sources = [
            Path(config.DATA_DIR) / "sberbank_help_documents.json",
        ]
        for json_file in json_sources:
            if json_file.exists():
                json_chunks.extend(load_json_documents(str(json_file)))
        faq_lookup.update_faq_documents(json_chunks)
        
        # 3. Объединяем все чанки
        all_chunks = pdf_chunks + json_chunks
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_chunks)})")
            
        # 4. Создаём векторное хранилище
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None
