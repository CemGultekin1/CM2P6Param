from typing import List

from models.load import load_modelsdict


class ModelsArray:
    def __init__(self,models:List[str]):
        self.modelsdict = load_modelsdict()
        self.models = models