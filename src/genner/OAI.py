import ast
import json
import re
from typing import Any, List, Tuple, cast, Optional

from openai import OpenAI
from result import Result, Ok, Err, UnwrapError

from src.typing.message import Message
from .Base import Genner
from dataclasses import dataclass, field

from typing import Dict, TypedDict, Any
from typing import Dict, NamedTuple

@dataclass
class PList:
    messages: List[Message] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.messages)

    def __add__(self, other: "PList") -> "PList":
        return PList(messages=self.messages + other.messages)

    def __repr__(self) -> str:
        messages_repr = pformat(self.messages)
        return f"PList(\n\tmessages=[\n\t\t{messages_repr}\n\t\t]\n)"

class OAIConfig(NamedTuple):
    name: str = "OpenAI"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.5

class OAIGenner(Genner):
    def __init__(self, client: OpenAI, config: OAIConfig):
        super().__init__("oai")

        self.client = client
        self.config = config

    def plist_completion(self, messages: PList) -> Result[str, str]:
        try:
            print(self.config.model)
            response = self.client.chat.completions.create(
                model=self.config.model.strip(),
                # response_format={"type": "json_object"},
                messages=cast(Any, messages),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            assert isinstance(response.choices[0].message.content, str)

            return Ok(response.choices[0].message.content)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return Err(
                "OAIGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    def generate_code(
        self, messages: PList
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            extracted_code = self.extract_code(raw_response).unwrap()
            return Ok((extracted_code, raw_response))
        except UnwrapError as e:
            error_message = (
                "OAIGenner.generate_code: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "OAIGenner.generate_code: Unexpected error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            # Extract code from the response
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            assert code_match is not None, "`code_match` is None"

            code_string = code_match.group(1)
            assert code_string is not None, "`code_string` is None"

            return Ok(code_string)
        except Exception as e:
            return Err(
                "OAIGenner.extract_code: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    def generate_list(
        self, messages: PList
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            processed_list = self.extract_list(raw_response).unwrap()
            return Ok((processed_list, raw_response))
        except UnwrapError as e:
            error_message = (
                "OAIGenner.generate_list: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "OAIGenner.generate_list: Unexpected error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        try:
            # Remove markdown code block markers and "json" label
            json_str = response.replace("```json", "").replace("```", "").strip()

            expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

            for key in expected_keys:
                if key in json_str:
                    processed_list = json.loads(json_str)[key]
                    break
            else:
                return Err(
                    f"No strategies found in the response, expected keys: {expected_keys}"
                )

            # Validate types
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(
                isinstance(item, str) for item in processed_list
            ), "All items in `processed_list` must be strings"

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "OAIGenner.extract_list: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )