"""
Registry system for BayesPrompt to manage models, datasets, metrics, and losses.
"""

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict = {}

    def register(self, name: str = None):
        def _register(obj):
            key = name if name is not None else obj.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._module_dict[key] = obj
            return obj
        return _register

    def get(self, name: str):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not found in {self._name}")
        return self._module_dict[name]

    @property
    def registered_names(self):
        return list(self._module_dict.keys())

MODELS = Registry("Models")
DATASETS = Registry("Datasets")
LOSSES = Registry("Losses")
METRICS = Registry("Metrics")
