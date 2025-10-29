import asyncio
import logging
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command

from .config import Settings
from .llm import TutorLLM
from .memory import InMemoryDialogStore

FALLBACK_REPLY = "Sorry, I had trouble answering. Please try again."
WELCOME_MESSAGE = (
    "Hi! I'm your English tutor bot. Ask me anything about English, "
    "and I'll do my best to help."
)
HELP_MESSAGE = "Ask questions about grammar, vocabulary, or practice dialogues."


async def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("bot.main")

    bot = Bot(token=settings.telegram_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    llm = TutorLLM(settings=settings)
    memory = InMemoryDialogStore(limit=6)

    @dp.message(Command("start"))
    async def handle_start(message: types.Message) -> None:
        memory.reset(message.from_user.id)
        await message.answer(WELCOME_MESSAGE)

    @dp.message(Command("help"))
    async def handle_help(message: types.Message) -> None:
        await message.answer(HELP_MESSAGE)

    @dp.message(F.text)
    async def handle_message(message: types.Message) -> None:
        user = message.from_user
        user_id = user.id if user else message.chat.id
        user_text = message.text or ""

        prior_history = memory.get(user_id)
        llm_history = [*prior_history, ("user", user_text)]

        try:
            reply = await llm.reply(user_message=user_text, history=llm_history)
            memory.add(user_id, "user", user_text)
            memory.add(user_id, "assistant", reply)
        except Exception:
            logger.exception("LLM request failed")
            memory.add(user_id, "user", user_text)
            reply = FALLBACK_REPLY
        await message.answer(reply)

    logger.info(
        "Starting polling",
        extra={
            "openrouter_model": settings.openrouter_model,
            "context_limit": memory.limit,
        },
    )

    try:
        await dp.start_polling(bot)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Bot shutdown requested")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
