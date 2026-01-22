from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from loguru import logger
from ollama import ChatResponse, Client, chat
from result import Err, Ok, Result, UnwrapError, is_err

from src.helper import timeout
from src.typing.message import Message

from .Base import Genner
from .config import (
    OllamaConfig,
)


class OllamaGenner(Genner):
    def __init__(self, config: OllamaConfig, identifier: str):
        super().__init__(identifier)
        self.client = Client(
            host=config.endpoint,
        )

        self.config = config

    def plist_completion(self, messages: List[Message]) -> Result[str, str]:
        try:
            with timeout(300):
                response: ChatResponse = self.client.chat(self.config.model, messages)

            if response.message.content is None:
                return Err(
                    "OllamaGenner.plist_completion: No content in response,\n"  #
                    f"`self.config.model`: {self.config.model}\n"
                    f"`messages`: \n{messages}"
                )

            return Ok(response.message.content)
        except TimeoutError as e:
            return Err(
                "OllamaGenner.plist_completion: Timeout error,\n"  #
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )
        except Exception as e:
            return Err(
                "OllamaGenner.plist_completion: Unexpected error,\n"  #
                f"`self.config`: {self.config}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    def generate_code(
        self, messages: List[Message], wrap_code: bool = True
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            extracted_code = self.extract_code(raw_response).unwrap()

            return Ok((extracted_code, raw_response))
        except UnwrapError as e:
            err_msg_prefix = "OllamaGenner.generate_code"
            if raw_response is None:  # Error in plist_completion
                error_message = (
                    f"{err_msg_prefix}: Plist completion failed,\n"
                    f"`self.config.name`: {self.config.name}\n"
                    f"`self.config.model`: {self.config.model}\n"
                    f"`e.result.err()`: \n{e.result.err()}\n"
                )
            else:  # Error in extract_code
                error_message = (
                    f"{err_msg_prefix}: Code extraction failed,\n"
                    f"`self.config.name`: {self.config.name}\n"
                    f"`self.config.model`: {self.config.model}\n"
                    f"`raw_response`: \n{raw_response}\n"
                    f"`e.result.err()`: \n{e.result.err()}\n"
                )
            return Err((error_message, raw_response))
        except Exception as e:
            err_msg_prefix = "OllamaGenner.generate_code"
            context_info = (
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
            )
            if raw_response is None:  # Error likely in plist_completion or before it
                error_message = (
                    f"{err_msg_prefix}: Unexpected error during plist completion,\n"
                    f"{context_info}"
                    f"`e`: \n{e}\n"
                )
            else:  # Error likely in extract_code
                error_message = (
                    f"{err_msg_prefix}: Unexpected error during code extraction,\n"
                    f"{context_info}"
                    f"`raw_response`: \n{raw_response}\n"
                    f"`e`: \n{e}\n"
                )
            return Err((error_message, raw_response))

    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            # raw_response here is guaranteed to be a string due to successful unwrap above
            processed_list = self.extract_list(raw_response).unwrap()  # type: ignore
            return Ok((processed_list, raw_response))  # type: ignore
        except UnwrapError as e:
            err_msg_prefix = "OllamaGenner.generate_list"
            if raw_response is None:  # Error in plist_completion
                error_message = (
                    f"{err_msg_prefix}: Plist completion failed,\n"
                    f"`self.config.name`: {self.config.name}\n"
                    f"`self.config.model`: {self.config.model}\n"
                    f"`e.result.err()`: \n{e.result.err()}\n"
                )
            else:  # Error in extract_list
                error_message = (
                    f"{err_msg_prefix}: List extraction failed,\n"
                    f"`self.config.name`: {self.config.name}\n"
                    f"`self.config.model`: {self.config.model}\n"
                    f"`raw_response`: \n{raw_response}\n"
                    f"`e.result.err()`: \n{e.result.err()}\n"
                )
            return Err((error_message, raw_response))
        except Exception as e:
            err_msg_prefix = "OllamaGenner.generate_list"
            context_info = (
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
            )
            if raw_response is None:  # Error likely in plist_completion or before it
                error_message = (
                    f"{err_msg_prefix}: Unexpected error during plist completion,\n"
                    f"{context_info}"
                    f"`e`: \n{e}\n"
                )
            else:  # Error likely in extract_list
                error_message = (
                    f"{err_msg_prefix}: Unexpected error during list extraction,\n"
                    f"{context_info}"
                    f"`raw_response`: \n{raw_response}\n"
                    f"`e`: \n{e}\n"
                )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        return Ok(response)

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        return Ok(response.splitlines())
