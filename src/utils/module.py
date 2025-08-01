from abc import ABC, abstractmethod
from typing import Optional


class Module(ABC):
    def __init__(
        self,
        path_to_model: Optional[str] = None,
    ):
        self.path_to_model = path_to_model


    @abstractmethod
    def load_pretrained_model(
        self,
        pretrained_path: str,
    ): ...

    @abstractmethod
    def predict(self, x): ...
