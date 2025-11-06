import logging
import re
import base64
from datetime import time
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from openai import APIError, InternalServerError, NotFoundError
from llm import get_transaction_response_text, get_transaction_response_image
from models import Transaction
from config import config
from stt import convert_to_wav, transcribe_audio
import tempfile
import os
import shutil

logger = logging.getLogger(__name__)
router = Router()

# Глобальные словари для хранения данных
chat_conversations: dict[int, list[dict]] = {}
transactions: dict[int, list[Transaction]] = {}

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000


def _compute_balance(user_id: int) -> float:
    return sum(
        t.amount if t.type.value == "income" else -t.amount
        for t in transactions.get(user_id, [])
    )


def _format_balance(balance: float) -> str:
    return f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"


def _remove_last_transaction(user_id: int, query: str | None = None) -> tuple[bool, str | None]:
    """Remove last transaction optionally matching a query.

    Returns (removed, removed_category_or_desc).
    """
    user_tx = transactions.get(user_id, [])
    if not user_tx:
        return False, None

    if query:
        q = query.strip().lower()
        stem4 = q[:4]
        for idx in range(len(user_tx) - 1, -1, -1):
            t = user_tx[idx]
            hay = f"{t.category} {t.description}".lower()
            words = re.findall(r"[a-zа-яё0-9]+", hay)
            direct = q in hay
            prefix = any(w.startswith(stem4) for w in words) if len(stem4) >= 3 else False
            short = (len(q) >= 5 and q[:-1] in hay)
            if direct or prefix or short:
                removed = user_tx.pop(idx)
                transactions[user_id] = user_tx
                return True, removed.category or removed.description or q
        return False, None
    else:
        removed = user_tx.pop()
        transactions[user_id] = user_tx
        return True, removed.category or (removed.description[:30] if removed.description else None)

@router.message(Command("start"))
async def cmd_start(message: Message):
    chat_id = message.chat.id
    logger.info(f"User {chat_id} started the bot")
    
    # Очищаем историю и транзакции для данного чата
    chat_conversations[chat_id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
    ]
    transactions[chat_id] = []
    
    await message.answer(
        "Привет! Я персональный финансовый советник.\n\n"
        "Я могу:\n"
        "• Извлекать транзакции из ваших сообщений\n"
        "• Вести учет доходов и расходов\n"
        "• Предоставлять советы по управлению финансами\n\n"
        "Используйте /start для начала нового диалога и очистки истории."
    )

@router.message(Command("balance"))
async def cmd_balance(message: Message):
    chat_id = message.chat.id
    logger.info(f"Balance requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "💵 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Расчет баланса, доходов и расходов
    total_income = sum(t.amount for t in user_transactions if t.type.value == "income")
    total_expense = sum(t.amount for t in user_transactions if t.type.value == "expense")
    balance = total_income - total_expense
    
    # Статистика по категориям
    category_stats: dict[str, float] = {}
    for t in user_transactions:
        category = t.category
        if category not in category_stats:
            category_stats[category] = 0.0
        if t.type.value == "income":
            category_stats[category] += t.amount
        else:
            category_stats[category] -= t.amount
    
    # Форматирование отчета
    report_lines = [
        "💵 **Отчет о балансе**\n",
        f"📊 Баланс: {balance:.2f} руб.",
        f"💰 Доходы: {total_income:.2f} руб.",
        f"💸 Расходы: {total_expense:.2f} руб.",
        f"\n📈 Всего транзакций: {len(user_transactions)}",
        "\n**Статистика по категориям:**"
    ]
    
    # Сортируем категории по сумме (от большей к меньшей)
    sorted_categories = sorted(category_stats.items(), key=lambda x: abs(x[1]), reverse=True)
    for category, amount in sorted_categories:
        sign = "💰" if amount > 0 else "💸"
        report_lines.append(f"{sign} {category}: {amount:+.2f} руб.")
    
    await message.answer("\n".join(report_lines))

@router.message(Command("transactions"))
async def cmd_transactions(message: Message):
    chat_id = message.chat.id
    logger.info(f"Transactions list requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "📋 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Сортируем транзакции по дате (от новых к старым)
    sorted_transactions = sorted(user_transactions, key=lambda t: (t.date, t.time or time(0, 0)), reverse=True)
    
    # Форматирование списка транзакций
    report_lines = [
        f"📋 **Все транзакции** ({len(user_transactions)} шт.)\n"
    ]
    
    for i, t in enumerate(sorted_transactions, 1):
        # Форматирование даты и времени
        date_str = t.date.strftime("%d.%m.%Y")
        time_str = f" {t.time.strftime('%H:%M')}" if t.time else ""
        
        # Знак и тип транзакции
        sign = "💰" if t.type.value == "income" else "💸"
        type_str = "Доход" if t.type.value == "income" else "Расход"
        
        # Форматирование суммы
        amount_str = f"{t.amount:.2f}".rstrip('0').rstrip('.')
        
        # Описание (если есть)
        desc_str = f"\n   {t.description}" if t.description else ""
        
        report_lines.append(
            f"{i}. {sign} **{type_str}** {amount_str} руб.\n"
            f"   📅 {date_str}{time_str}\n"
            f"   🏷️ {t.category}{desc_str}"
        )
    
    # Если транзакций много, разбиваем на несколько сообщений (Telegram лимит ~4096 символов)
    report_text = "\n\n".join(report_lines)
    if len(report_text) > 4000:
        # Разбиваем на части
        parts = []
        current_part = [report_lines[0]]  # Заголовок
        current_length = len(report_lines[0])
        
        for line in report_lines[1:]:
            line_length = len(line) + 2  # +2 для "\n\n"
            if current_length + line_length > 4000:
                parts.append("\n\n".join(current_part))
                current_part = [line]
                current_length = len(line)
            else:
                current_part.append(line)
                current_length += line_length
        
        if current_part:
            parts.append("\n\n".join(current_part))
        
        # Отправляем части
        for part in parts:
            await message.answer(part)
    else:
        await message.answer(report_text)

@router.message(lambda message: message.photo or (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")))
async def handle_image(message: Message):
    chat_id = message.chat.id
    
    logger.info(f"Image received from {chat_id}")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE}
        ]
    
    try:
        # Определяем источник изображения
        if message.photo:
            # Берем самое большое изображение
            photo = message.photo[-1]
            file_info = await message.bot.get_file(photo.file_id)
        elif message.document:
            file_info = await message.bot.get_file(message.document.file_id)
        else:
            await message.answer("Не удалось обработать изображение.")
            return
        
        # Скачиваем изображение
        file_buffer = await message.bot.download_file(file_info.file_path)
        image_bytes = file_buffer.getvalue()
        
        # Конвертируем в base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        # Получаем ответ LLM с structured output
        response = await get_transaction_response_image(image_base64, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for image from {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from image for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from image for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount 
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем изображение в историю как текстовое описание (для контекста)
        chat_conversations[chat_id].append(
            {"role": "user", "content": "[Изображение: чек/скриншот]"}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError, NotFoundError) as e:
        logger.error(f"LLM API error for image from {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        if "image input" in error_message.lower() or "404" in error_message or "not found" in error_message.lower():
            await message.answer(
                "Извините, используемая модель не поддерживает обработку изображений.\n\n"
                "Для работы с изображениями необходимо использовать vision-модель, например:\n"
                "• meta-llama/llama-3.2-11b-vision-instruct (OpenRouter)\n"
                "• llama3.2-vision (Ollama)\n\n"
                "Измените MODEL в файле .env на одну из этих моделей."
            )
        else:
            await message.answer(
                "Извините, произошла ошибка на стороне провайдера LLM при обработке изображения. "
                "Пожалуйста, попробуйте еще раз через несколько секунд."
            )
    except Exception as e:
        logger.error(f"Error processing image from {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке изображения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

@router.message(lambda message: message.voice or message.audio)
async def handle_voice(message: Message):
    chat_id = message.chat.id
    logger.info(f"Voice/audio message received from {chat_id}")

    # Инициализируем историю, если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
        ]

    tmpdir = tempfile.mkdtemp(prefix="tg-voice-")
    src_path = os.path.join(tmpdir, "input.ogg")
    wav_path = os.path.join(tmpdir, "output.wav")

    try:
        # Получаем файл
        if message.voice:
            file_info = await message.bot.get_file(message.voice.file_id)
        elif message.audio:
            file_info = await message.bot.get_file(message.audio.file_id)
        else:
            await message.answer("Не удалось обработать голосовое сообщение.")
            return

        file_buffer = await message.bot.download_file(file_info.file_path)
        with open(src_path, "wb") as f:
            f.write(file_buffer.getvalue())

        # Конвертация в WAV 16kHz mono
        await convert_to_wav(src_path, wav_path)

        # Транскрибация в текст
        text = await transcribe_audio(wav_path)
        if not text:
            await message.answer("Не удалось распознать речь на аудио.")
            return

        logger.info(f"Transcribed voice from {chat_id}: {text[:120]}...")

        # Подготовка истории для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []

        # Запрашиваем structured output по распознанному тексту
        response = await get_transaction_response_text(text, message_history)

        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)

        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount
            for t in transactions.get(chat_id, [])
        )

        # Собираем ответ
        answer_text = f"🗣️ Распознал текст:\n{text}\n\n" + response.answer
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"

        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."

        # Обновляем историю диалога
        chat_conversations[chat_id].append({"role": "user", "content": text})
        chat_conversations[chat_id].append({"role": "assistant", "content": response.answer})

        await message.answer(answer_text)

    except FileNotFoundError as e:
        logger.error(f"ffmpeg not found for {chat_id}: {e}")
        await message.answer(
            "Не удалось конвертировать аудио. Убедитесь, что установлен ffmpeg (macOS: brew install ffmpeg, Ubuntu/Debian: apt-get install ffmpeg)."
        )
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM/STT API error for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Извините, произошла ошибка при распознавании речи/обработке текста. Попробуйте еще раз."
        )
    except Exception as e:
        logger.error(f"Error processing voice/audio from {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке голосового сообщения. Попробуйте еще раз."
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    # Проверяем длину сообщения
    if len(message.text) > MAX_MESSAGE_LENGTH:
        await message.answer(
            f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
            f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
        )
        return
    
    chat_id = message.chat.id
    last_message = message.text
    
    logger.info(f"Message from {chat_id}: {last_message[:100]}...")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
        ]
    
    # Получаем историю сообщений без системного промпта для контекста
    message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []

    # Быстрые намерения до вызова LLM
    normalized = last_message.lower().strip()

    # 1) Удаление последней транзакции (по категории/описанию или без уточнения)
    remove_intent = re.search(r"\b(удали|удалить|удалишь|убери|убрать|отмени|отменить)\b", normalized)
    last_token = re.search(r"\bпоследн[а-я]+\b", normalized)
    if remove_intent and last_token:
        # Попробуем извлечь уточнение после слова 'последн*'
        m = re.search(r"последн[а-я]+\s+(.+)$", normalized)
        query = m.group(1).strip() if m else None
        # Если уточнение похоже на слово 'транзакц*', не используем его
        if query and re.match(r"^транзакц", query):
            query = None

        removed, key = _remove_last_transaction(chat_id, query)
        balance = _compute_balance(chat_id)
        if removed:
            key_text = f" о {key}" if key else ""
            answer_text = (
                f"Я убрал последнюю запись{key_text}.\n\n"
                f"🗑️ Удалена 1 транзакция\n"
                f"💵 Баланс: {_format_balance(balance)} руб."
            )
        else:
            answer_text = (
                "Не нашёл подходящую транзакцию для удаления.\n\n"
                f"ℹ️ Транзакции не удалены\n"
                f"💵 Баланс: {_format_balance(balance)} руб."
            )

        chat_conversations[chat_id].append({"role": "user", "content": last_message})
        chat_conversations[chat_id].append({"role": "assistant", "content": answer_text})
        await message.answer(answer_text)
        return

    # 2) Явный доход: "пришла зарплата 54321", "зарплата 120000"
    salary_match = re.search(r"\b(пришл[аио]|зарплат[аыуеы]|получил[аи]?)\b.*?(\d+[\d\s.,]*)", normalized)
    if salary_match:
        from datetime import date
        from models import Transaction, TransactionType, TransactionFrequency

        raw_amount = salary_match.group(2)
        amt = raw_amount.replace(" ", "").replace(",", ".")
        amt = re.sub(r"[^0-9.].*$", "", amt)
        try:
            amount = float(amt)
        except ValueError:
            amount = None

        if amount and amount > 0:
            tx = Transaction(
                date=date.today(),
                time=None,
                type=TransactionType.INCOME,
                amount=amount,
                frequency=TransactionFrequency.PERIODIC,
                category="зарплата",
                description="Зарплата"
            )
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].append(tx)

            balance = _compute_balance(chat_id)
            answer_text = (
                f"Записал ваш доход 'зарплата' в размере {amount:g} рублей.\n\n"
                f"✅ Найдено и сохранено 1 транзакция\n"
                f"💵 Баланс: {_format_balance(balance)} руб."
            )
            chat_conversations[chat_id].append({"role": "user", "content": last_message})
            chat_conversations[chat_id].append({"role": "assistant", "content": "Записал ваш доход: зарплата."})
            await message.answer(answer_text)
            return

    try:
        # Получаем ответ LLM с structured output (извлечение транзакций только из последнего сообщения)
        response = await get_transaction_response_text(last_message, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from message: '{last_message}' for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount 
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем сообщение пользователя в историю
        chat_conversations[chat_id].append(
            {"role": "user", "content": last_message}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Извините, произошла ошибка на стороне провайдера LLM. "
            "Пожалуйста, попробуйте еще раз через несколько секунд."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )
