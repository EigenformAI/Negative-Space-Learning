from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from result import Result

from src.typing.message import Message
from src.typing.alias import RawResponse, ParsedCode, ParsedList


class Genner(ABC):
    def __init__(self, identifier: str):
        self.identifier = identifier

    @abstractmethod
    def plist_completion(self, messages: List[Message]) -> Result[RawResponse, str]:
        pass

    @abstractmethod
    def generate_code(
        self, messages: List[Message]
    ) -> Result[Tuple[ParsedCode, RawResponse], Tuple[str, Optional[RawResponse]]]:
        pass

    @abstractmethod
    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[ParsedList, RawResponse], Tuple[str, Optional[RawResponse]]]:
        pass

    @staticmethod
    @abstractmethod
    def extract_code(response: RawResponse) -> Result[ParsedCode, str]:
        pass

    @staticmethod
    @abstractmethod
    def extract_list(response: RawResponse) -> Result[ParsedList, str]:
        pass
