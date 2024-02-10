import asyncio
import logging
import os
import pickle
import sys
from enum import IntEnum
from pathlib import Path
from random import randint

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.markdown import hbold

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

import translators as ts

from api.api_advices import APIAdvices
from api.api_currency import APIConverter
from api.api_google_search import google
from text_normalizer import text_normalization

api_rates = APIConverter()
api_advices = APIAdvices()

BASE_DIR = Path(__file__).resolve().parent

data_agent_path = os.path.join(BASE_DIR, "data_agent.pickle")
model_path = os.path.join(BASE_DIR, "model.pickle")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pickle")


try:
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    data_agent = pickle.load(open(data_agent_path, 'rb'))
except FileNotFoundError:
    print("Model and vector are not found. Please, train model first with help of 'model_training.py' file.")


class Settings(BaseSettings):
    bot_token: SecretStr
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


class Mode(IntEnum):
    RATES = 1
    CONVERSION = 2


class CurrencyCallback(CallbackData, prefix='currency'):
    mode: Mode
    value: str | int | None


config = Settings()
bot = Bot(
        token=config.bot_token.get_secret_value(),
        parse_mode=ParseMode.HTML,
    )
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Привет, {hbold(message.from_user.full_name)}!")


@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    """
    This handler receives messages with `/help` command
    """
    await message.answer(f"Нужна помощь, {hbold(message.from_user.full_name)}?")


@dp.message()
async def message_handler(message: Message) -> None:
    translated_text = text_normalization(ts.translate_text(message.text, to_language="en", translator="bing"))
    if result := model.predict(vectorizer.transform([translated_text])):
        print(result)
        key_word = result[0]
        if key_word == "guide me":
            await message.answer(api_advices.get_random_advice())
        elif key_word == "answer":
            examples = data_agent.get(key_word).get('examples')
            request_text = text_normalization(translated_text)
            for ex in examples:
                request_text = request_text.replace(ex, '')

            result = google(request_text)
            if result:
                result_text = "Результаты поиска:\n\n"
                for result_item in result:
                    if (res_text := result_item.get('text')) and (res_link := result_item.get('url')):
                        result_text += f"{res_text} [{res_link}]\n\n"
                await message.answer(result_text)
            else:
                await message.answer("Не могу найти ответ на ваш вопрос, попробуйте задать его по другому")
        elif key_word == "rates":
            buttons = []
            for key in api_rates.currency_list.keys():
                currency = api_rates.currency_list.get(key)
                buttons.append(
                    [
                        InlineKeyboardButton(
                            text=currency,
                            callback_data=CurrencyCallback(
                                mode=Mode.RATES,
                                value=currency,
                            ).pack(),
                        )
                    ]
                )
            markup = InlineKeyboardMarkup(inline_keyboard=buttons)
            await message.answer("Выберите валюту", reply_markup=markup)
        else:
            if responses := data_agent.get(key_word).get('responses'):
                print(responses)
                resp_len = len(responses)
                if response_res := data_agent.get(key_word).get('responses')[randint(0, resp_len - 1)]:
                    await message.answer(ts.translate_text(response_res, to_language="ru", translator="bing"))
    else:
        await message.answer("Жду нормальное сообщение...")


@dp.callback_query(CurrencyCallback.filter(F.mode == Mode.RATES))
async def callback_rates_inline(query: CallbackQuery, callback_data: CurrencyCallback):
    base = callback_data.value
    result_text = ""
    for quote in api_rates.currencies.values():
        if quote == base:
            continue
        result_text += f"1 {base} = {APIConverter.get_rate(base, quote)} {quote}\n"
    await query.message.answer(result_text)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())

