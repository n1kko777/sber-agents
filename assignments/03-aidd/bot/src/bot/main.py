import asyncio
import os

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode

FIXED_REPLY = "Hello! I'm your English tutor bot prototype."


async def handle_message(message: types.Message) -> None:
    await message.answer(FIXED_REPLY)


async def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    bot = Bot(token=token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    dp.message.register(handle_message, F.text)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
