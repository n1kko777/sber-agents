import asyncio
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode

from .config import Settings
from .llm import TutorLLM

FALLBACK_REPLY = "Sorry, I had trouble answering. Please try again."


async def main() -> None:
    settings = Settings.from_env()
    bot = Bot(token=settings.telegram_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    llm = TutorLLM(settings=settings)

    @dp.message(F.text)
    async def handle_message(message: types.Message) -> None:
        try:
            reply = await llm.reply(message.text or "")
        except Exception:
            reply = FALLBACK_REPLY
        await message.answer(reply)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
