import json
import re
import ast
from typing import List, Tuple

from result import Result, Ok, Err

from .OAI import OAIGenner
from .config import QwenConfig


class QwenVllmGenner(OAIGenner):
    def __init__(self, client, config: QwenConfig):
        super().__init__(client, config)

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            # Try standard python code block first
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            
            if code_match is not None:
                code_string = code_match.group(1)
                if code_string is not None:
                    return Ok(code_string)
            
            # Fallback: try any code block
            regex_pattern = r"```(?:python)?\n?([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            
            if code_match is not None:
                code_string = code_match.group(1)
                if code_string is not None:
                    return Ok(code_string)
            
            # Last resort: look for python-like code without code blocks
            if "import subprocess" in response and "def main():" in response:
                # Extract everything from import to end
                start_idx = response.find("import subprocess")
                if start_idx != -1:
                    code_section = response[start_idx:]
                    # Clean up any trailing text after the last function
                    if "main()" in code_section:
                        end_idx = code_section.rfind("main()") + len("main()")
                        return Ok(code_section[:end_idx])
            
            return Err("QwenGenner.extract_code: No valid code found")
            
        except Exception as e:
            return Err(
                "QwenGenner.extract_code: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        # Try JSON parsing first
        try:
            # Remove markdown code block markers and "json" label
            json_str = response.replace("```json", "").replace("```", "").strip()
            
            # Extract only the JSON part (from first { to matching })
            json_start = json_str.find('{')
            if json_start == -1:
                raise ValueError("No JSON object found")
            
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if brace_count != 0:
                raise ValueError("Unmatched braces in JSON")
            
            clean_json = json_str[json_start:json_end]

            expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

            parsed_json = json.loads(clean_json)
            for key in expected_keys:
                if key in parsed_json:
                    processed_list = parsed_json[key]
                    break
            else:
                return Err(
                    "QwenGenner.extract_list: No matching strategy keys found in the response, "
                    f"`response`: \n{response}\n"
                    f"`expected_keys`: \n{expected_keys}\n"
                    f"`parsed_json`: \n{parsed_json}\n"
                )

            # Validate types
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All items in `processed_list` must be strings"
            )

            return Ok(processed_list)
        except Exception as e:
            pass_1_exception = e

        try:
            match = re.search(r"```(.*)```", response, re.DOTALL)

            if not match:
                return Err(
                    "QwenGenner.extract_list: No backticks (```) found in the response, "
                    f"`response`: \n{response}\n"
                )

            # Extract the content inside the backticks (group 1 of the match)
            content_inside_backticks = match.group(1).strip()

            processed_list = []
            # Split the extracted content into individual lines
            lines = content_inside_backticks.strip().split("\n")

            for line in lines:
                # Check if the line starts with a hyphen
                if line.strip().startswith("-"):
                    # Remove the leading hyphen and any surrounding whitespace
                    list_item = line.strip().lstrip("- ").strip()
                    processed_list.append(list_item)

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "QwenGenner.extract_list: Failed to parse response as JSON and also failed to extract strategies from backticks, "
                f"`pass_1_exception`: \n{pass_1_exception}\n"
                f"`pass_2_exception`: \n{e}\n"
                f"`response`: \n{response}\n"
            )
