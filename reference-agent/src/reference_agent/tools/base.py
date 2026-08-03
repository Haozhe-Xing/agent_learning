from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for agent tools.

    Subclasses must declare `name`, `description`, `parameters` (a JSON
    schema) and implement `run`.
    """

    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs) -> str:
        ...

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
