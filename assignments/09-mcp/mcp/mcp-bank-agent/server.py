#!/usr/bin/env python3
"""
Bank Agent MCP Server

Предоставляет три инструмента для банковского агента:
1. search_products - поиск актуальных продуктов банка (вклады, кредиты, карты)
2. currency_converter - конвертация валют по курсам ЦБ РФ
3. early_repayment_calculator - расчёт выгоды от досрочного погашения кредита

Транспорт: streamable-http (HTTP MCP server)
Порт: 8000 (по умолчанию для FastMCP)
"""
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal
import requests
from pydantic import Field

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-bank-agent")

# Path to the products database
PRODUCTS_DB_PATH = Path(__file__).parent / "data" / "bank_products.json"

# CBR API endpoint
CBR_API_URL = "https://www.cbr-xml-daily.ru/latest.js"


def load_products() -> list[dict]:
    """Загрузка продуктов банка из JSON файла."""
    try:
        if not PRODUCTS_DB_PATH.exists():
            logger.error(f"Products database not found at {PRODUCTS_DB_PATH}")
            return []
        
        with open(PRODUCTS_DB_PATH, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        logger.info(f"Loaded {len(products)} products from database")
        return products
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return []


def filter_products(
    products: list[dict],
    product_type: str | None = None,
    keyword: str | None = None,
    min_amount: int | None = None,
    max_amount: int | None = None,
    min_rate: float | None = None,
    max_rate: float | None = None,
    currency: str | None = None
) -> list[dict]:
    """
    Фильтрация продуктов по параметрам
    
    Использует list comprehension для простоты (следуя принципу KISS).
    """
    filtered = products
    
    # Фильтр по типу продукта
    if product_type:
        filtered = [p for p in filtered if p.get('product_type') == product_type]
    
    # Поиск по ключевому слову (в названии и описании)
    if keyword:
        keyword_lower = keyword.lower()
        filtered = [
            p for p in filtered
            if keyword_lower in p.get('name', '').lower() or 
               keyword_lower in p.get('description', '').lower()
        ]
    
    # Фильтр по минимальной сумме
    if min_amount is not None:
        filtered = [p for p in filtered if p.get('amount_min', 0) <= min_amount]
    
    # Фильтр по максимальной сумме
    if max_amount is not None:
        filtered = [p for p in filtered if p.get('amount_max', float('inf')) >= max_amount]
    
    # Фильтр по минимальной ставке
    if min_rate is not None:
        filtered = [p for p in filtered if p.get('rate_max', 0) >= min_rate]
    
    # Фильтр по максимальной ставке
    if max_rate is not None:
        filtered = [p for p in filtered if p.get('rate_min', float('inf')) <= max_rate]
    
    # Фильтр по валюте
    if currency:
        filtered = [p for p in filtered if currency in p.get('currency', '')]
    
    return filtered


def format_products(products: list[dict], limit: int = 10) -> str:
    """
    Форматирование списка продуктов для агента
    
    Возвращает топ-N продуктов с основной информацией.
    """
    if not products:
        return "Продукты не найдены по заданным критериям."
    
    # Ограничиваем количество результатов
    products = products[:limit]
    
    result = f"Найдено {len(products)} продукт(ов):\n\n"
    
    for i, product in enumerate(products, 1):
        result += f"**{i}. {product.get('name')}**\n"
        result += f"   Описание: {product.get('description')}\n"
        
        # Ставка (для вкладов и кредитов)
        rate_min = product.get('rate_min', 0)
        rate_max = product.get('rate_max', 0)
        if rate_min > 0 or rate_max > 0:
            if rate_min == rate_max:
                result += f"   Ставка: {rate_min}% годовых\n"
            else:
                result += f"   Ставка: от {rate_min}% до {rate_max}% годовых\n"
        
        # Сумма
        amount_min = product.get('amount_min', 0)
        amount_max = product.get('amount_max', 0)
        if amount_min > 0 or amount_max > 0:
            if amount_max > 0:
                result += f"   Сумма: от {amount_min:,} до {amount_max:,} {product.get('currency', 'RUB')}\n"
            else:
                result += f"   Сумма: от {amount_min:,} {product.get('currency', 'RUB')}\n"
        
        # Срок
        term = product.get('term_months', '')
        if term:
            result += f"   Срок: {term} месяцев\n"
        
        # Особенности
        features = product.get('features', [])
        if features:
            result += f"   Особенности: {', '.join(features)}\n"
        
        result += "\n"
    
    return result


def get_exchange_rates() -> dict:
    """
    Получение курсов валют от ЦБ РФ
    
    API возвращает курсы относительно рубля (base: RUB).
    Например: {"USD": 0.0124} означает 1 RUB = 0.0124 USD (или 1 USD ≈ 80.6 RUB)
    """
    try:
        response = requests.get(CBR_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('rates', {})
    except requests.RequestException as e:
        logger.error(f"Error fetching exchange rates: {e}")
        return {}


def convert_currency(
    from_currency: str,
    to_currency: str,
    amount: float | None,
    rates: dict
) -> tuple[float | None, str]:
    """
    Конвертация валюты через рубль
    
    Логика:
    - RUB → другая валюта: amount * rates[to_currency]
    - другая валюта → RUB: amount / rates[from_currency]
    - валюта1 → валюта2: amount / rates[from] * rates[to] (через рубли)
    
    Returns:
        (converted_amount, formatted_string)
    """
    if not rates:
        return None, "Не удалось получить курсы валют от ЦБ РФ"
    
    # Проверка поддержки валют
    if from_currency != "RUB" and from_currency not in rates:
        return None, f"Валюта {from_currency} не поддерживается"
    
    if to_currency != "RUB" and to_currency not in rates:
        return None, f"Валюта {to_currency} не поддерживается"
    
    # Одинаковые валюты
    if from_currency == to_currency:
        rate_str = f"1 {from_currency} = 1 {to_currency}"
        if amount:
            return amount, f"{amount:,.2f} {from_currency} = {amount:,.2f} {to_currency}"
        return 1.0, rate_str
    
    # Конвертация через рубль
    if from_currency == "RUB":
        # RUB → другая валюта
        rate = rates[to_currency]
        rate_str = f"1 RUB = {rate:.6f} {to_currency} (или 1 {to_currency} ≈ {1/rate:.2f} RUB)"
        if amount:
            converted = amount * rate
            return converted, f"{amount:,.2f} RUB = {converted:,.2f} {to_currency}\n\nТекущий курс: {rate_str}"
        return rate, rate_str
    
    elif to_currency == "RUB":
        # другая валюта → RUB
        rate = rates[from_currency]
        rate_str = f"1 {from_currency} = {1/rate:.2f} RUB (или 1 RUB = {rate:.6f} {from_currency})"
        if amount:
            converted = amount / rate
            return converted, f"{amount:,.2f} {from_currency} = {converted:,.2f} RUB\n\nТекущий курс: {rate_str}"
        return 1/rate, rate_str
    
    else:
        # валюта1 → валюта2 (через рубль)
        rate_from = rates[from_currency]  # from → RUB
        rate_to = rates[to_currency]      # RUB → to
        rate = (1 / rate_from) * rate_to  # итоговый курс from → to
        
        rate_str = f"1 {from_currency} = {rate:.4f} {to_currency}"
        if amount:
            converted = amount * rate
            return converted, f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency}\n\nТекущий курс: {rate_str}"
        return rate, rate_str


def calculate_monthly_payment(
    principal: float,
    monthly_rate: float,
    months: int
) -> float:
    """
    Расчет аннуитетного платежа.

    Используется формула: A = P * r / (1 - (1 + r)^-n)
    """
    if months <= 0:
        raise ValueError("Срок кредита должен быть больше нуля")
    
    if monthly_rate == 0:
        return principal / months
    
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** (-months))


def amortize(
    balance: float,
    monthly_payment: float,
    monthly_rate: float,
    months_limit: int
) -> tuple[float, int, float, float]:
    """
    Проводит расчёт помесячно с заданным платежом.

    Returns:
        (остаток, месяцев прошло, выплаченные проценты, выплаченная сумма)
    """
    total_interest = 0.0
    total_paid = 0.0
    months_passed = 0
    
    for _ in range(months_limit):
        if balance <= 1e-6:
            break
        
        months_passed += 1
        interest = balance * monthly_rate
        principal_payment = monthly_payment - interest
        
        if principal_payment <= 0:
            raise ValueError("Ежемесячный платеж меньше начисленных процентов")
        
        if principal_payment > balance:
            principal_payment = balance
            payment_fact = interest + principal_payment
        else:
            payment_fact = monthly_payment
        
        balance -= principal_payment
        total_interest += interest
        total_paid += payment_fact
    
    return balance, months_passed, total_interest, total_paid


def amortize_full(
    balance: float,
    monthly_payment: float,
    monthly_rate: float,
    max_months: int = 1200
) -> tuple[int, float, float]:
    """
    Доводит расчёт до полного погашения или сообщает, что платеж не покрывает проценты.
    """
    total_interest = 0.0
    total_paid = 0.0
    months_used = 0
    
    while balance > 1e-6 and months_used < max_months:
        months_used += 1
        interest = balance * monthly_rate
        principal_payment = monthly_payment - interest
        
        if principal_payment <= 0:
            raise ValueError("Ежемесячный платеж меньше начисленных процентов")
        
        if principal_payment > balance:
            principal_payment = balance
            payment_fact = interest + principal_payment
        else:
            payment_fact = monthly_payment
        
        balance -= principal_payment
        total_interest += interest
        total_paid += payment_fact
    
    if balance > 1e-6:
        raise ValueError("Не удалось погасить кредит за разумный срок — проверьте параметры")
    
    return months_used, total_interest, total_paid


# Create FastMCP server
mcp = FastMCP("mcp-bank-agent", dependencies=["requests>=2.31.0"])


@mcp.tool(
    name="search_products",
    description="Универсальный поиск актуальных продуктов банка (вклады, кредиты, карты, счета) с гибкой фильтрацией",
)
async def search_products(
    product_type: Annotated[
        Literal["deposit", "credit", "debit_card", "credit_card", "account"] | None,
        Field(
            description="Тип продукта для фильтрации",
        )
    ] = None,
    keyword: Annotated[
        str | None,
        Field(
            description="Ключевое слово для поиска в названии и описании продукта",
            min_length=2,
            max_length=100,
            examples=["вклад", "кредит", "карта", "кешбэк"]
        )
    ] = None,
    min_amount: Annotated[
        int | None,
        Field(
            description="Минимальная сумма (ищет продукты доступные от этой суммы)",
            ge=0,
            examples=[10000, 50000, 100000]
        )
    ] = None,
    max_amount: Annotated[
        int | None,
        Field(
            description="Максимальная сумма (ищет продукты доступные до этой суммы)",
            ge=0,
            examples=[1000000, 5000000]
        )
    ] = None,
    min_rate: Annotated[
        float | None,
        Field(
            description="Минимальная процентная ставка (для вкладов и кредитов)",
            ge=0,
            le=100,
            examples=[10.0, 15.0, 20.0]
        )
    ] = None,
    max_rate: Annotated[
        float | None,
        Field(
            description="Максимальная процентная ставка (для вкладов и кредитов)",
            ge=0,
            le=100,
            examples=[15.0, 20.0, 25.0]
        )
    ] = None,
    currency: Annotated[
        Literal["RUB", "USD", "EUR"] | None,
        Field(
            description="Валюта продукта"
        )
    ] = None
) -> str:
    """
    Поиск актуальных банковских продуктов с фильтрацией
    
    Этот инструмент ищет текущие продукты банка с актуальными ставками и условиями.
    В отличие от rag_search (статические PDF), здесь динамические данные о продуктах.
    
    Args:
        product_type: Тип продукта (вклад, кредит, карта, счёт)
        keyword: Поиск по ключевому слову
        min_amount: Минимальная сумма
        max_amount: Максимальная сумма
        min_rate: Минимальная ставка
        max_rate: Максимальная ставка
        currency: Валюта
    
    Returns:
        Форматированный список найденных продуктов (топ-10)
    """
    logger.info(f"search_products called with: type={product_type}, keyword={keyword}, "
                f"amount={min_amount}-{max_amount}, rate={min_rate}-{max_rate}, currency={currency}")
    
    # Загружаем продукты
    products = load_products()
    if not products:
        return "Не удалось загрузить базу продуктов банка"
    
    # Фильтруем
    filtered = filter_products(
        products,
        product_type=product_type,
        keyword=keyword,
        min_amount=min_amount,
        max_amount=max_amount,
        min_rate=min_rate,
        max_rate=max_rate,
        currency=currency
    )
    
    # Форматируем результат
    return format_products(filtered)


@mcp.tool(
    name="currency_converter",
    description="Конвертация валют по актуальным курсам ЦБ РФ с поддержкой всех основных валют",
)
async def currency_converter(
    from_currency: Annotated[
        Literal["RUB", "USD", "EUR", "CNY", "GBP", "CHF", "JPY", "TRY"],
        Field(
            description="Исходная валюта для конвертации"
        )
    ] = "USD",
    to_currency: Annotated[
        Literal["RUB", "USD", "EUR", "CNY", "GBP", "CHF", "JPY", "TRY"],
        Field(
            description="Целевая валюта для конвертации"
        )
    ] = "RUB",
    amount: Annotated[
        float | None,
        Field(
            description="Сумма для конвертации (если не указана, вернется только курс)",
            ge=0,
            examples=[100, 1000, 10000]
        )
    ] = None
) -> str:
    """
    Конвертация валют по актуальным курсам ЦБ РФ
    
    Поддерживает конвертацию между любыми валютами (не только с рублями).
    Данные обновляются ежедневно ЦБ РФ.
    
    Args:
        from_currency: Исходная валюта
        to_currency: Целевая валюта
        amount: Сумма для конвертации (опционально)
    
    Returns:
        Результат конвертации с текущим курсом
    """
    logger.info(f"currency_converter called: {amount} {from_currency} -> {to_currency}")
    
    # Получаем актуальные курсы
    rates = get_exchange_rates()
    
    # Конвертируем
    converted_amount, result_str = convert_currency(from_currency, to_currency, amount, rates)
    
    if converted_amount is None:
        return result_str  # Сообщение об ошибке
    
    return result_str


@mcp.tool(
    name="early_repayment_calculator",
    description="Калькулятор досрочного погашения кредита: уменьшение срока или ежемесячного платежа",
)
async def early_repayment_calculator(
    loan_amount: Annotated[
        float,
        Field(
            description="Сумма кредита",
            gt=0,
            examples=[1_000_000, 2_500_000]
        )
    ],
    annual_rate: Annotated[
        float,
        Field(
            description="Годовая процентная ставка в процентах",
            gt=0,
            le=100,
            examples=[12.5, 18.9]
        )
    ],
    term_months: Annotated[
        int,
        Field(
            description="Срок кредита в месяцах",
            ge=3,
            le=600,
            examples=[36, 60, 120]
        )
    ],
    early_payment: Annotated[
        float,
        Field(
            description="Сумма единовременного досрочного платежа",
            gt=0,
            examples=[100_000, 300_000]
        )
    ],
    month_number: Annotated[
        int,
        Field(
            description="Месяц, в который планируется внести досрочный платеж (1 — первый месяц)",
            ge=1,
            examples=[3, 6, 12]
        )
    ] = 1,
    strategy: Annotated[
        Literal["reduce_term", "reduce_payment"],
        Field(
            description="Стратегия: reduce_term — оставляем платеж и сокращаем срок, reduce_payment — оставляем срок и уменьшаем платеж"
        )
    ] = "reduce_term",
    currency: Annotated[
        str,
        Field(
            description="Валюта для отображения результата",
            examples=["RUB", "USD"]
        )
    ] = "RUB"
) -> str:
    """
    Рассчитывает эффект от досрочного погашения аннуитетного кредита.
    
    Логика:
    - Выполняем стандартные платежи до выбранного месяца
    - Вносим досрочный платеж
    - Стратегия reduce_term: сохраняем платеж, рассчитываем новый срок
    - Стратегия reduce_payment: сохраняем срок, пересчитываем платеж
    """
    logger.info("early_repayment_calculator called")
    currency = currency.upper()
    
    try:
        if month_number > term_months:
            return "Номер месяца досрочного платежа не может быть больше срока кредита"
        
        monthly_rate = annual_rate / 12 / 100
        base_payment = calculate_monthly_payment(loan_amount, monthly_rate, term_months)
        base_total_interest = base_payment * term_months - loan_amount
        
        # Платежи до момента досрочного погашения
        balance_after_regular, months_passed, interest_before, payments_before = amortize(
            loan_amount,
            base_payment,
            monthly_rate,
            min(month_number, term_months)
        )
        
        if months_passed < month_number and balance_after_regular <= 1e-6:
            return "Кредит будет погашен раньше выбранного месяца, досрочный платеж не требуется."
        
        balance_before_extra = balance_after_regular
        balance_after_extra = max(balance_before_extra - early_payment, 0)
        
        total_interest = interest_before
        total_paid = payments_before + early_payment
        
        if balance_after_extra <= 1e-6:
            interest_saved = max(base_total_interest - total_interest, 0)
            return (
                "Долг полностью погашен за счёт досрочного платежа.\n\n"
                f"Базовый ежемесячный платёж: {base_payment:,.2f} {currency}\n"
                f"Проценты по графику: {base_total_interest:,.2f} {currency}\n"
                f"Фактически заплатите процентов: {total_interest:,.2f} {currency}\n"
                f"Экономия на процентах: {interest_saved:,.2f} {currency}\n"
                f"Всего выплатите: {total_paid:,.2f} {currency}"
            )
        
        if strategy == "reduce_payment":
            remaining_months = max(term_months - months_passed, 1)
            new_payment = calculate_monthly_payment(balance_after_extra, monthly_rate, remaining_months)
            
            interest_after = new_payment * remaining_months - balance_after_extra
            interest_after = max(interest_after, 0.0)
            
            total_interest += interest_after
            total_paid += new_payment * remaining_months
            total_months = months_passed + remaining_months
            
            interest_saved = max(base_total_interest - total_interest, 0.0)
            payment_diff = base_payment - new_payment
            
            return (
                "Стратегия: уменьшаем ежемесячный платёж, срок остаётся прежним.\n\n"
                f"Базовый платёж: {base_payment:,.2f} {currency}\n"
                f"Новый платёж после досрочного: {new_payment:,.2f} {currency} "
                f"(изменение: {'-' if payment_diff >= 0 else '+'}{abs(payment_diff):,.2f})\n"
                f"Проценты по графику: {base_total_interest:,.2f} {currency}\n"
                f"Проценты с учётом досрочного: {total_interest:,.2f} {currency}\n"
                f"Экономия на процентах: {interest_saved:,.2f} {currency}\n"
                f"Всего выплатите: {total_paid:,.2f} {currency}\n"
                f"Осталось платить месяцев: {remaining_months} (общий срок: {total_months} мес)"
            )
        
        # strategy == reduce_term
        months_after, interest_after, payments_after = amortize_full(
            balance_after_extra,
            base_payment,
            monthly_rate,
            max_months=term_months * 2
        )
        
        total_interest += interest_after
        total_paid += payments_after
        total_months = months_passed + months_after
        months_saved = max(term_months - total_months, 0)
        
        interest_saved = max(base_total_interest - total_interest, 0.0)
        
        return (
            "Стратегия: сокращаем срок, платёж остаётся прежним.\n\n"
            f"Базовый платёж: {base_payment:,.2f} {currency}\n"
            f"Остаток перед досрочным: {balance_before_extra:,.2f} {currency}\n"
            f"Остаток после досрочного: {balance_after_extra:,.2f} {currency}\n"
            f"Новый срок: {total_months} мес (экономия {months_saved} мес)\n"
            f"Проценты по графику: {base_total_interest:,.2f} {currency}\n"
            f"Проценты с учётом досрочного: {total_interest:,.2f} {currency}\n"
            f"Экономия на процентах: {interest_saved:,.2f} {currency}\n"
            f"Всего выплатите: {total_paid:,.2f} {currency}"
        )
    
    except ValueError as e:
        logger.error(f"early_repayment_calculator error: {e}")
        return f"Ошибка в расчёте: {e}"


if __name__ == "__main__":
    logger.info("Starting Bank Agent MCP Server...")
    logger.info(f"Products database: {PRODUCTS_DB_PATH}")
    logger.info(f"Currency API: {CBR_API_URL}")
    
    # Проверяем наличие базы продуктов
    if not PRODUCTS_DB_PATH.exists():
        logger.error(f"Products database not found at {PRODUCTS_DB_PATH}")
        logger.error("Please create data/bank_products.json before starting the server")
        exit(1)
    
    # Получаем порт из переменной окружения (по умолчанию 8000)
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Server will be available at: http://localhost:{port}/mcp")
    
    # Запускаем сервер
    mcp.run(transport="streamable-http")
