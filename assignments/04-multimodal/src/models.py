from pydantic import BaseModel, Field
from datetime import date, time
from enum import Enum
from typing import Optional

class TransactionType(str, Enum):
    INCOME = "income"      # доход
    EXPENSE = "expense"    # расход

class TransactionFrequency(str, Enum):
    DAILY = "daily"           # повседневные
    PERIODIC = "periodic"     # периодические
    ONE_TIME = "one_time"     # разовые

class Transaction(BaseModel):
    date: date                           # дата транзакции
    time: Optional[time] = None            # время (опционально)
    type: TransactionType                # доход/расход
    amount: float = Field(gt=0)          # сумма (строго положительная)
    frequency: TransactionFrequency       # тип (повседневные, периодические, разовые)
    category: str                        # категория (продукты, рестораны, такси и т.д.)
    description: str = ""                # описание транзакции (подробная информация о товарах, услугах, источнике, контрагенте и т.п.)

class TransactionResponse(BaseModel):
    transactions: list[Transaction]  # список транзакций (всегда должен быть, пустой [] если не найдено)
    answer: str                     # текстовый ответ пользователю (обязателен)

