import logging
from typing import Optional, Dict, Any
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import config
import rag

logger = logging.getLogger(__name__)

# Глобальные инициализированные метрики
_ragas_metrics = None
_ragas_run_config = None

def create_ragas_embeddings():
    """
    Фабрика для создания RAGAS embeddings по провайдеру из конфига
    Поддерживает: openai, huggingface
    """
    provider = config.RAGAS_EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI embeddings: {config.RAGAS_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(model=config.RAGAS_EMBEDDING_MODEL)
    
    elif provider == "huggingface":
        logger.info(f"Creating RAGAS HuggingFace embeddings: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
        return HuggingFaceEmbeddings(
            model_name=config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={'device': config.RAGAS_HUGGINGFACE_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"Unknown RAGAS embedding provider: {provider}. Use 'openai' or 'huggingface'")

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Настройка LLM и embeddings для RAGAS (фиксированные модели для единообразной оценки)
    langchain_llm = ChatOpenAI(model=config.RAGAS_LLM_MODEL, temperature=0)
    langchain_embeddings = create_ragas_embeddings()
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    # Инициализируем метрики
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = ragas_llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = ragas_embeddings
        run_config = RunConfig()
        metric.init(run_config)
    
    # Настройки для выполнения
    run_config = RunConfig(
        max_workers=4,
        max_wait=180,
        max_retries=3
    )
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    
    logger.info(f"✓ RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
    logger.info(f"✓ RAGAS LLM: {config.RAGAS_LLM_MODEL}")
    logger.info(f"✓ RAGAS Embedding Provider: {config.RAGAS_EMBEDDING_PROVIDER}")
    if config.RAGAS_EMBEDDING_PROVIDER == "openai":
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
    
    return _ragas_metrics, _ragas_run_config

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Проверка существования датасета в LangSmith
    
    Args:
        dataset_name: имя датасета
    
    Returns:
        True если датасет существует
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def evaluate_dataset(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Главная функция evaluation RAG системы
    
    По образцу референсного ноутбука (раздел 5.2):
    1. Запуск эксперимента в LangSmith с blocking=False и сбор данных
    2. RAGAS batch evaluation
    3. Загрузка метрик как feedback в LangSmith
    
    Args:
        dataset_name: имя датасета (по умолчанию из конфига)
    
    Returns:
        dict с результатами evaluation
    """
    if not config.LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set. Cannot run evaluation.")
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
    
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        # Используем существующую RAG цепочку
        # Передаем только вопрос (без истории для evaluation)
        from langchain_core.messages import HumanMessage
        result = rag.get_rag_chain().invoke({"messages": [HumanMessage(content=question)]})
        
        return {
            "answer": result["answer"],
            "documents": result["documents"]
        }
    
    # Собираем данные во время выполнения evaluate
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    run_ids = []
    
    # evaluate() с blocking=False возвращает итератор
    for result in client.evaluate(
        target,
        data=dataset_name,
        evaluators=[],
        experiment_prefix="rag-evaluation",
        metadata={
            "approach": "RAGAS batch evaluation + LangSmith feedback",
            "model": config.MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        blocking=False,
    ):
        run = result["run"]
        example = result["example"]
        
        # Получаем данные
        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        documents = run.outputs.get("documents", [])
        contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        ground_truth = example.outputs.get("answer", "") if example else ""
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation
    ragas_result = evaluate(
        ragas_dataset,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
    )
    
    ragas_df = ragas_result.to_pandas()
    
    logger.info("RAGAS evaluation completed")
    
    # Вычисляем средние значения метрик
    metrics_summary = {}
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            avg_score = ragas_df[metric.name].mean()
            metrics_summary[metric.name] = avg_score
            logger.info(f"  {metric.name}: {avg_score:.3f}")
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                client.create_feedback(
                    run_id=run_id,
                    key=metric.name,
                    score=float(score),
                    comment=f"RAGAS metric: {metric.name}"
                )
    
    logger.info(f"Feedback uploaded ({len(run_ids)} runs)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

