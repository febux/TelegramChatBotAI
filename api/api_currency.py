import requests
import lxml.html

import yaml

from conf.constants import BASE_DIR


# класс интерфейса приложения по конвертации валют
class APIConverter:
    def __init__(self, currencies_file: str = f'{BASE_DIR}/api/api_data/currencies.yaml'):
        with open(currencies_file, 'r', encoding='UTF-8') as file:
            self.currencies = yaml.safe_load(file)

    # статический метод конвертации валют, полученных с сайта
    # входные параметры берутся из вне
    @staticmethod
    def get_rate(base, quote, amount=1):
        req_curr = requests.get(f'https://freecurrencyrates.com/ru/convert-{base}-{quote}#{amount}')
        req_curr_content = lxml.html.document_fromstring(req_curr.content)
        exchange_rate = req_curr_content.xpath('/html/body/main/div/div[2]/div[1]/div[1]'
                                               '/div[2]/div[2]/input/@value')

        return format(float(amount) * float(exchange_rate[0]), '.2f')

    # метод получения списка доступных валют
    def get_available_list_currency(self):
        available_list_currency = '\n - ' + '\n - '.join(self.currencies.keys())
        return available_list_currency

    # метод получения списка доступных валют
    @property
    def currency_list(self):
        return self.currencies

    # метод установки списка доступных валют
    def set_list_currency(self, filename: str = 'currencies.yaml'):
        with open(filename, 'r', encoding='UTF-8') as file:
            self.currencies = yaml.safe_load(file)
