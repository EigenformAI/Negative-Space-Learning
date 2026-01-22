import ast
import re
from typing import Any, List, Tuple, cast, Optional

import requests
from loguru import logger
from result import Result, Ok, Err, UnwrapError

from .config import DreamConfig
from src.typing.message import Message

from .Base import Genner


class DreamGenner(Genner):
    def __init__(self, config: DreamConfig):
        super().__init__("dream")
        self.config = config

    def plist_completion(self, messages: List[Message]) -> Result[str, str]:
        url = f"{self.config.base_url.rstrip('/')}/generate"

        payload = {
            "messages": messages,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "steps": self.config.steps,
            "alg": self.config.alg,
            "timeout": self.config.timeout,
        }

        try:
            r = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout,
            )

            r.raise_for_status()

            json_payload = cast(dict[str, Any], r.json())
            text_response = cast(str, json_payload["result"])

            return Ok(text_response)
        except Exception as e:
            return Err(
                "DreamGenner.plist_completion: Unexpected error,"
                f"`url`: {url}\n"
                f"`self.config`: \n{str(self.config)}\n"
                f"`e`: \n{e}\n"
            )

    def generate_code(
        self, messages: List[Message]
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            extracted_code = self.extract_code(raw_response).unwrap()
            return Ok((extracted_code, raw_response))
        except UnwrapError as e:
            error_message = (
                "DreamGenner.generate_code: Unwrap error,\n"
                f"`self.config`: {self.config}\n"  # Note: DreamConfig doesn't have .name or .model directly like ClaudeConfig
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "DreamGenner.generate_code: Unexpected error,\n"
                f"`self.config`: {self.config}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        if "```python" not in response or "```" not in response:
            return Ok(response.strip())

        try:
            pattern = r"```python\n([\s\S]*?)```"
            match = re.search(pattern, response, re.DOTALL)

            assert match is not None, "Code block not found"

            code_str = match.group(1)
            assert isinstance(code_str, str), "Code block is not a string"

            return Ok(code_str)
        except Exception as e:
            return Err(
                "DreamGenner.extract_code: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            extracted_list = self.extract_list(raw_response).unwrap()
            return Ok((extracted_list, raw_response))
        except UnwrapError as e:
            error_message = (
                "DreamGenner.generate_list: Unwrap error,\n"
                f"`self.config`: {self.config}\n"  # Note: DreamConfig doesn't have .name or .model directly
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "DreamGenner.generate_list: Unexpected error,\n"
                f"`self.config`: {self.config}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        try:
            start, end = response.index("["), response.rindex("]") + 1
            list_str = response[start:end]
            parsed = cast(List[str], ast.literal_eval(list_str))

            assert all(isinstance(item, str) for item in parsed), (
                "All items must be strings"
            )

            return Ok(parsed)
        except Exception as e:
            return Err(
                "DreamGenner.extract_list: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )
