"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
import os
import re
from langchain_core.tools import tool
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None
import rag

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


def _get_tavily_client() -> "TavilyClient":
    """Создает Tavily клиент и валидирует конфигурацию."""
    if TavilyClient is None:
        raise RuntimeError(
            "Пакет tavily-python не установлен. Добавьте его в зависимости перед использованием Tavily."
        )
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден TAVILY_API_KEY в окружении.")
    return TavilyClient(api_key=api_key)


def _parse_number(value: str) -> float | None:
    """Преобразует строку с числом в float, поддерживает запятые."""
    try:
        normalized = value.replace(",", ".").strip()
        return float(normalized)
    except Exception:
        return None


def _parse_rates_from_text(text: str) -> dict[str, dict[str, float]]:
    """
    Извлекает курсы валют из текста Tavily (ожидаем блоки с покупкой/продажей).
    
    Возвращает словарь вида {"USD": {"buy": 00.00, "sell": 00.00}, ...}
    """
    if not text:
        return {}
    
    normalized = re.sub(r"\s+", " ", text)
    normalized_lower = normalized.lower()
    currency_aliases = {
        "USD": [r"USD", r"Доллар[^A-Za-z0-9]{0,3}США", r"Доллара?"],
        "EUR": [r"EUR", r"Евро"],
    }
    buy_labels = {"покупка", "buy", "продать"}  # банк покупает валюту, пользователь продает
    sell_labels = {"продажа", "sell", "купить"}  # банк продает валюту, пользователь покупает
    
    rates: dict[str, dict[str, float]] = {}
    for code, aliases in currency_aliases.items():
        alias_pattern = "|".join(aliases)
        window_match = re.search(rf"(?:{alias_pattern}).{{0,320}}", normalized, flags=re.IGNORECASE)
        window_text = normalized if window_match is None else normalized[window_match.start():window_match.end()]
        
        buy: float | None = None
        sell: float | None = None
        
        # Ищем метки "Купить/Продать/Покупка/Продажа" перед числами
        for match in re.finditer(r"(покупка|продажа|buy|sell|купить|продать)[^\d]{0,12}(\d+[.,]\d+)", window_text, flags=re.IGNORECASE):
            label = match.group(1).lower()
            number = _parse_number(match.group(2))
            if number is None:
                continue
            if label in buy_labels and buy is None:
                buy = number
            elif label in sell_labels and sell is None:
                sell = number
            if buy is not None and sell is not None:
                break
        
        # Фолбек: берем первые два числа около названия валюты
        if buy is None or sell is None:
            numbers = [n for n in re.findall(r"(\d+[.,]\d+)", window_text)]
            if len(numbers) >= 2:
                buy = _parse_number(numbers[0])
                sell = _parse_number(numbers[1])
        
        # Санити-чек значений (отсекаем явно некорректные 1-5 руб)
        def _valid(rate: float | None) -> bool:
            return rate is not None and 10.0 <= rate <= 300.0
        
        if _valid(buy) and _valid(sell):
            rates[code] = {"buy": buy, "sell": sell}
    
    return rates


def _fetch_sberbank_rates() -> dict[str, dict[str, float]]:
    """
    Получает курсы валют через Tavily с сайта Сбербанк Онлайн (вкладка СБОЛ).
    
    Используем include_domains чтобы ограничить поиск только сайтом Сбербанка.
    """
    client = _get_tavily_client()
    queries = [
        "Курсы валют СберБанк Онлайн СБОЛ покупка продажа USD EUR https://www.sberbank.ru/ru/quotes/currencies?tab=sbol&currency=USD&currency=EUR",
        "курс доллара евро сегодня Сбербанк Онлайн купить продать",
    ]
    
    def _search(query: str, include_sber: bool):
        kwargs = {
            "query": query,
            "search_depth": "advanced",
            "max_results": 8,
            "include_answer": True,
            "include_raw_content": True,
        }
        if include_sber:
            kwargs["include_domains"] = ["www.sberbank.ru", "sberbank.ru"]
        return client.search(**kwargs)
    
    def _try_parse(search_result: dict) -> dict[str, dict[str, float]]:
        results = search_result.get("results", []) or []
        
        # 1) приоритетно парсим результаты с доменом sberbank.ru
        for res in results:
            url = (res.get("url") or "").lower()
            if "sberbank.ru" not in url:
                continue
            for key in ("raw_content", "content"):
                text = res.get(key) or ""
                rates = _parse_rates_from_text(text)
                if rates:
                    return rates
        
        # 2) парсим все остальные результаты
        for res in results:
            for key in ("raw_content", "content"):
                text = res.get(key) or ""
                rates = _parse_rates_from_text(text)
                if rates:
                    return rates
        
        # 3) answer от Tavily
        answer = search_result.get("answer", "") or ""
        rates = _parse_rates_from_text(answer)
        if rates:
            return rates
        
        # 4) агрегируем все тексты
        texts = []
        for res in results:
            for key in ("raw_content", "content"):
                val = res.get(key)
                if val:
                    texts.append(val)
        combined = " ".join(texts)
        return _parse_rates_from_text(combined)
    
    for query in queries:
        # Сначала строго по домену Сбербанк
        result = _search(query, include_sber=True)
        rates = _try_parse(result)
        if rates:
            return rates
        
        # Фолбек: шире поиск без ограничения доменов
        result = _search(query, include_sber=False)
        rates = _try_parse(result)
        if rates:
            return rates
    
    logger.warning("Не удалось получить курсы валют через Tavily.")
    return {}


def _convert_amount(amount: float, from_currency: str, to_currency: str, rates: dict[str, dict[str, float]]) -> tuple[float, list[str]]:
    """
    Конвертирует сумму используя купля/продажа относительно RUB.
    
    Логика:
    - RUB -> FX: делим на курс продажи FX (покупаем валюту)
    - FX -> RUB: умножаем на курс покупки FX (банк покупает валюту)
    - FX1 -> FX2: через промежуточный RUB (покупка FX1 -> RUB -> продажа FX2)
    """
    steps: list[str] = []
    if from_currency == to_currency:
        return amount, steps
    
    if from_currency == "RUB":
        sell_rate = rates[to_currency]["sell"]
        result = amount / sell_rate
        steps.append(f"RUB -> {to_currency} по продаже {sell_rate:.4f}")
        return result, steps
    
    if to_currency == "RUB":
        buy_rate = rates[from_currency]["buy"]
        result = amount * buy_rate
        steps.append(f"{from_currency} -> RUB по покупке {buy_rate:.4f}")
        return result, steps
    
    buy_rate = rates[from_currency]["buy"]
    rub_amount = amount * buy_rate
    sell_rate = rates[to_currency]["sell"]
    result = rub_amount / sell_rate
    steps.extend([
        f"{from_currency} -> RUB по покупке {buy_rate:.4f}",
        f"RUB -> {to_currency} по продаже {sell_rate:.4f}",
    ])
    return result, steps


@tool
def currency_rates() -> str:
    """
    Возвращает текущие курсы покупки/продажи USD и EUR по Сбербанк Онлайн (вкладка СБОЛ).
    """
    try:
        rates = _fetch_sberbank_rates()
        if not rates:
            return "Не удалось получить курсы валют."
        
        parts = []
        for code in ("USD", "EUR"):
            if code in rates:
                parts.append(f"{code}: покупка {rates[code]['buy']:.2f}₽, продажа {rates[code]['sell']:.2f}₽")
        if not parts:
            return "Не удалось разобрать курсы валют."
        return "Актуальные курсы СБОЛ: " + "; ".join(parts)
    except Exception as e:
        logger.error(f"Error in currency_rates: {e}", exc_info=True)
        return "Не удалось получить курсы валют. Попробуйте позже."


@tool
def tavily_search(query: str) -> str:
    """
    Веб-поиск через Tavily (нужен TAVILY_API_KEY).
    
    Возвращает JSON с результатами (title, url, content) чтобы агент мог анализировать внешний контент.
    """
    try:
        client = _get_tavily_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
            })
        
        return json.dumps({"results": results}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in tavily_search: {e}", exc_info=True)
        return json.dumps({"results": []}, ensure_ascii=False)


@tool
def currency_converter(amount: float = 1.0, from_currency: str = "USD", to_currency: str = "RUB") -> str:
    """
    Конвертирует сумму из одной валюты в другую (USD, EUR, RUB) по курсам СБОЛ.
    
    Args:
        amount: Сумма для конвертации (по умолчанию 1)
        from_currency: Исходная валюта (USD, EUR, RUB) - по умолчанию USD
        to_currency: Целевая валюта (USD, EUR, RUB) - по умолчанию RUB
    
    Returns:
        Строка с результатом конвертации
    """
    try:
        if amount <= 0:
            return "Сумма должна быть положительной."
        
        from_cur = from_currency.upper()
        to_cur = to_currency.upper()
        supported = {"USD", "EUR", "RUB"}
        if from_cur not in supported or to_cur not in supported:
            return "Поддерживаемые валюты: USD, EUR, RUB."
        
        rates = _fetch_sberbank_rates()
        missing = [c for c in {from_cur, to_cur} if c != "RUB" and c not in rates]
        if missing:
            return f"Не удалось получить курсы: {', '.join(missing)}."
        
        converted, steps = _convert_amount(amount, from_cur, to_cur, rates)
        
        rate_info_parts = []
        for code, data in rates.items():
            rate_info_parts.append(f"{code}: покупка {data['buy']:.2f}₽ / продажа {data['sell']:.2f}₽")
        rate_info = "; ".join(rate_info_parts)
        
        details = f"Шаги конвертации: {'; '.join(steps)}." if steps else "Шаги конвертации: без изменений валюты."
        return (
            f"{amount:.2f} {from_cur} = {converted:.2f} {to_cur} по курсам Сбербанк Онлайн (СБОЛ).\n"
            f"{details}\n"
            f"Актуальные курсы: {rate_info}"
        )
    except Exception as e:
        logger.error(f"Error in currency_converter: {e}", exc_info=True)
        return "Не удалось выполнить конвертацию. Попробуйте позже."
