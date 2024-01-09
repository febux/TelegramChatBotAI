from random import randint

import yaml


class APIAdvices:
    def __init__(self, advices_file: str = 'advices.yaml'):
        with open(advices_file, 'r', encoding='UTF-8') as file:
            self.advices = yaml.safe_load(file)

    def get_random_advice(self):
        advice_number = randint(1, len(self.advices))
        return f"Жизненный совет №{advice_number}: {self.advices[advice_number]}"

    @property
    def advices_list(self):
        return self.advices

    def set_list_advices(self, filename: str = 'advices.yaml'):
        with open(filename, 'r', encoding='UTF-8') as file:
            self.advices = yaml.safe_load(file)
