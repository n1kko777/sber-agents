import argparse
import asyncio
import logging
from typing import List, Sequence

from langchain_core.messages import HumanMessage

from config import config
from indexer_with_json import reindex_all
import rag

DEFAULT_TEST_QUESTIONS = [
    "Как заказать карту?",
    "Какие документы нужны для получения карты?",
    "Сколько делают карту?",
    "Как активировать карту?",
    "Какие условия потребительского кредита?",
    "Какие виды вкладов есть?",
    "Какая процентная ставка по вкладу Сохраняй?",
]


def build_questions(cli_questions: Sequence[str] | None) -> List[str]:
    if cli_questions:
        return list(cli_questions)
    return DEFAULT_TEST_QUESTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Локальная проверка индекса и поиска по RAG без Telegram."
    )
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        dest="questions",
        help="Дополнительный вопрос (можно указать несколько раз).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.RETRIEVER_K,
        help="Сколько чанков показывать на каждый вопрос.",
    )
    parser.add_argument(
        "--ask-llm",
        action="store_true",
        help="Запросить полный ответ у RAG-цепочки (требует настроенный LLM).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Уровень логирования (например, INFO или DEBUG).",
    )
    return parser.parse_args()


def format_preview(text: str, limit: int = 320) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[:limit].rstrip() + "..."


async def run_llm_answers(questions: Sequence[str]) -> None:
    for question in questions:
        answer = await rag.rag_answer([HumanMessage(content=question)])
        print(f"\n🧠 Ответ RAG\nQ: {question}\nA: {answer}\n")


async def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())

    questions = build_questions(args.questions)
    vector_store = await reindex_all()
    if vector_store is None:
        raise SystemExit("Не удалось построить векторное хранилище. Проверьте данные.")

    doc_count = len(vector_store.store) if hasattr(vector_store, "store") else "unknown"
    print(f"Всего документов после индексации: {doc_count}")

    for question in questions:
        docs = vector_store.similarity_search(question, k=args.top_k)
        print(f"\n🔎 Вопрос: {question}")
        if not docs:
            print("  ✗ Ничего не найдено")
            continue
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            preview = format_preview(doc.page_content)
            print(f"  {idx}. {source} (стр. {page})")
            print(f"     {preview}")

    if args.ask_llm:
        rag.vector_store = vector_store
        rag.retriever = None
        if not rag.initialize_retriever():
            raise SystemExit("Не удалось инициализировать retriever.")
        await run_llm_answers(questions)


if __name__ == "__main__":
    asyncio.run(main())
