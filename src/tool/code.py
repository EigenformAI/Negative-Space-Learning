from loguru import logger
import ast
from typing import Tuple

from result import Result, Ok, Err


def is_valid_code_ast(code: str) -> Result[None, str]:
    try:
        ast.parse(code)
    except Exception as e:
        return Err(f"{e}")
    return Ok(None)


def is_valid_code_compiler(code: str) -> Result[None, str]:
    try:
        compile(code, "<string>", "exec")

        return Ok(None)
    except SyntaxError as e:
        return Err(f"{e}")


def validate_code_offline(
    code: str,
) -> Result[None, str]:
    """
    Validate the code.

    Args:
        code (str): The generated code to validate and run.

    Returns:
        Result[None, str]: Ok(None) if the code is valid, Err(str) if there is an error.
    """
    match is_valid_code_ast(code):
        case Ok(_):
            pass
        case Err(error_message):
            return Err(f"AST error: {error_message}")

    match is_valid_code_compiler(code):
        case Ok(_):
            pass
        case Err(error_message):
            return Err(f"Compiler error: {error_message}")

    return Ok(None)
