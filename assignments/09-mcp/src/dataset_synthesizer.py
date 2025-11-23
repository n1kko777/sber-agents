import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_sample_pdf_chunks(data_dir: str, samples_per_file: int = 2) -> List:
    """
    Загрузка PDF документов и выборка чанков для синтеза вопросов
    
    Args:
        data_dir: путь к директории с PDF файлами
        samples_per_file: количество чанков для выборки из каждого файла
    
    Returns:
        Список чанков с метаданными
    """
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_sampled_chunks = []
    
    for pdf_file in pdf_files:
        # Загружаем PDF
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        
        # Разбиваем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(pages)
        
        if not chunks:
            logger.warning(f"No chunks created from {pdf_file.name}")
            continue
        
        # Равномерная выборка чанков
        num_samples = min(samples_per_file, len(chunks))
        step = len(chunks) // num_samples if num_samples > 0 else 1
        sampled_chunks = [chunks[i * step] for i in range(num_samples)]
        
        all_sampled_chunks.extend(sampled_chunks)
        logger.info(f"Sampled {len(sampled_chunks)} chunks from {pdf_file.name}")
    
    return all_sampled_chunks

def load_json_qa_pairs(data_dir: str, samples_per_file: int = 2) -> List[Dict[str, Any]]:
    """
    Загрузка готовых Q&A пар из JSON файлов
    
    Args:
        data_dir: путь к директории с JSON файлами
        samples_per_file: количество Q&A пар для выборки из каждого файла
    
    Returns:
        Список Q&A пар в формате: {question, ground_truth, contexts, metadata}
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    all_qa_pairs = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Случайная выборка
        num_samples = min(samples_per_file, len(data))
        sampled_items = random.sample(data, num_samples)
        
        for item in sampled_items:
            qa_pair = {
                "question": item.get("question", ""),
                "ground_truth": item.get("answer", ""),
                "contexts": [item.get("full_text", item.get("answer", ""))],
                "metadata": {
                    "source": json_file.name,
                    "page": None,
                    "type": "from_json",
                    "category": item.get("category", "unknown"),
                    "url": item.get("url", "")
                }
            }
            all_qa_pairs.append(qa_pair)
        
        logger.info(f"Sampled {len(sampled_items)} Q&A pairs from {json_file.name}")
    
    return all_qa_pairs

def synthesize_qa_pairs_from_pdf(chunks: List, llm_model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """
    Синтез вопросов и ответов из PDF чанков через LLM
    
    Args:
        chunks: список чанков документов
        llm_model: модель для синтеза
    
    Returns:
        Список Q&A пар в формате: {question, ground_truth, contexts, metadata}
    """
    if not chunks:
        return []
    
    llm = ChatOpenAI(model=llm_model, temperature=0.7)
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
Ты эксперт по созданию вопросно-ответных пар для оценки RAG систем.
На основе предоставленного текста создай 1 разнообразный вопрос,
на который можно ответить используя этот текст.

Вопрос должен быть:
- Реалистичным (такие вопросы могут задать реальные пользователи)
- Конкретным (можно дать точный ответ на основе текста)
- На русском языке

Для вопроса также создай краткий точный ответ на основе текста.

ВАЖНО: Верни ТОЛЬКО валидный JSON без дополнительного текста:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}}
  ]
}}
        """),
        ("human", "Текст:\n{chunk_text}")
    ])
    
    qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content.strip()) < 100:
            logger.warning(f"Chunk {i} too short, skipping")
            continue
        
        try:
            response = llm.invoke(
                synthesis_prompt.format_messages(
                    chunk_text=chunk.page_content[:2000]
                )
            )
            
            # Парсим JSON ответ
            content = response.content.strip()
            
            # Извлекаем JSON из ответа (может быть обернут в markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            # Убираем возможные префиксы/суффиксы
            content = content.strip()
            if not content.startswith("{"):
                idx = content.find("{")
                if idx >= 0:
                    content = content[idx:]
            
            data = json.loads(content)
            
            for qa in data.get("qa_pairs", []):
                if "question" in qa and "answer" in qa:
                    qa_pairs.append({
                        "question": qa["question"],
                        "ground_truth": qa["answer"],
                        "contexts": [chunk.page_content],
                        "metadata": {
                            "source": chunk.metadata.get("source", "unknown"),
                            "page": chunk.metadata.get("page", -1),
                            "type": "synthesized"
                        }
                    })
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks, created {len(qa_pairs)} Q&A pairs")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for chunk {i}: {e}")
            logger.debug(f"LLM response: {content[:200]}...")
            continue
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue
    
    logger.info(f"Total synthesized {len(qa_pairs)} Q&A pairs from PDF")
    return qa_pairs

def create_dataset(data_dir: str, samples_per_file: int = 2) -> List[Dict[str, Any]]:
    """
    Создание полного датасета: синтез из PDF + готовые из JSON
    
    Args:
        data_dir: путь к директории с документами
        samples_per_file: количество примеров на файл
    
    Returns:
        Объединенный список Q&A пар
    """
    logger.info("Starting dataset creation...")
    
    # 1. Синтезируем из PDF
    logger.info("\n=== Synthesizing Q&A pairs from PDF ===")
    pdf_chunks = load_and_sample_pdf_chunks(data_dir, samples_per_file)
    pdf_qa_pairs = synthesize_qa_pairs_from_pdf(pdf_chunks)
    
    # 2. Загружаем готовые из JSON
    logger.info("\n=== Loading Q&A pairs from JSON ===")
    json_qa_pairs = load_json_qa_pairs(data_dir, samples_per_file)
    
    # 3. Объединяем
    all_qa_pairs = pdf_qa_pairs + json_qa_pairs
    
    logger.info(f"\n=== Dataset created ===")
    logger.info(f"PDF Q&A pairs (synthesized): {len(pdf_qa_pairs)}")
    logger.info(f"JSON Q&A pairs (from file): {len(json_qa_pairs)}")
    logger.info(f"Total Q&A pairs: {len(all_qa_pairs)}")
    
    return all_qa_pairs

def save_dataset(qa_pairs: List[Dict[str, Any]], filepath: str):
    """
    Сохранение датасета в JSON файл
    
    Args:
        qa_pairs: список Q&A пар
        filepath: путь для сохранения
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Total examples: {len(qa_pairs)}")

def upload_to_langsmith(dataset_path: str, dataset_name: str):
    """
    Загрузка датасета в LangSmith с проверкой дубликатов
    
    Args:
        dataset_path: путь к JSON файлу с датасетом
        dataset_name: имя датасета в LangSmith
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set. Cannot upload dataset.")
        return
    
    client = Client()
    
    # Загружаем датасет из файла
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # Проверяем существование датасета
    try:
        existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
        if existing_datasets:
            logger.warning(f"Dataset '{dataset_name}' already exists in LangSmith")
            response = input("Do you want to delete and recreate? (y/n): ")
            if response.lower() == 'y':
                for ds in existing_datasets:
                    client.delete_dataset(dataset_id=ds.id)
                    logger.info(f"Deleted existing dataset {ds.id}")
            else:
                logger.info("Upload cancelled")
                return
    except Exception as e:
        logger.debug(f"Dataset check: {e}")
    
    # Создаем датасет
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="RAG evaluation dataset from PDF synthesis and JSON Q&A pairs"
    )
    logger.info(f"Created dataset '{dataset_name}' (ID: {dataset.id})")
    
    # Загружаем примеры
    examples = []
    for qa in qa_pairs:
        example = {
            "inputs": {"question": qa["question"]},
            "outputs": {"answer": qa["ground_truth"]},
            "metadata": qa.get("metadata", {})
        }
        examples.append(example)
    
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        metadata=[ex["metadata"] for ex in examples]
    )
    
    logger.info(f"Uploaded {len(examples)} examples to LangSmith")
    logger.info(f"Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")

def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset synthesizer for RAG evaluation")
    parser.add_argument("--create", action="store_true", help="Create and save dataset locally")
    parser.add_argument("--upload", action="store_true", help="Upload existing dataset to LangSmith")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples per file")
    args = parser.parse_args()
    
    # Пути
    data_dir = config.DATA_DIR
    dataset_path = "datasets/06-rag-qa-dataset.json"
    dataset_name = config.LANGSMITH_DATASET
    
    # Создание датасета
    if args.create:
        logger.info("=== Creating dataset ===")
        qa_pairs = create_dataset(data_dir, samples_per_file=args.samples)
        save_dataset(qa_pairs, dataset_path)
    
    # Загрузка в LangSmith
    if args.upload:
        logger.info("\n=== Uploading to LangSmith ===")
        upload_to_langsmith(dataset_path, dataset_name)
    
    # Если ничего не указано
    if not args.create and not args.upload:
        parser.print_help()
        logger.error("\nError: Specify at least one action: --create or --upload")

if __name__ == "__main__":
    main()

