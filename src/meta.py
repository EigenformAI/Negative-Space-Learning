import functools
import inspect
import re
from typing import Callable, Any, Dict, TypeVar, Generic, Optional, TypedDict, ParamSpec

import ast
import textwrap
import difflib
import warnings

from pydantic import BaseModel


class WrapperCallReturnObject(BaseModel):
    formatted_prompt: str
    fn_name: str


# Type parameters for the original function's signature
P = ParamSpec("P")
R_original = TypeVar("R_original") # Return type of the original function


class PromptWrapper(Generic[P, R_original]):
    _prompt_template: str
    _original_func: Callable[P, R_original]
    # __name__, __qualname__, etc., are set by functools.update_wrapper
    # __annotations__ will also be copied if present on the original function

    def __init__(self, original_func_param: Callable[P, R_original], prompt_template: str):
        self._prompt_template = prompt_template
        self._original_func = original_func_param
        functools.update_wrapper(self, original_func_param)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> WrapperCallReturnObject:
        # Bind arguments to the original function's signature
        # to correctly get argument names and values for formatting.
        sig = inspect.signature(self._original_func)
        bound_arguments = sig.bind(*args, **kwargs) # P.args and P.kwargs are passed here
        bound_arguments.apply_defaults()
        call_args = bound_arguments.arguments.copy()

        returned_from_func = self._original_func(*args, **kwargs)

        args_for_formatting = call_args.copy()
        if isinstance(returned_from_func, dict):
            args_for_formatting.update(returned_from_func)
        elif returned_from_func is not None:
            print(
                f"Warning: Function '{self._original_func.__name__}' returned a non-dict, non-None value "
                f"({type(returned_from_func)}). This return value is not directly used for "
                "updating prompt formatting arguments unless it's a dictionary."
            )

        # Calculate template_placeholders here as it depends on self._prompt_template
        template_placeholders = set(re.findall(r"\{(\w+)\}", self._prompt_template))
        available_keys_for_formatting = set(args_for_formatting.keys())
        
        if not template_placeholders.issubset(available_keys_for_formatting):
            missing_keys = template_placeholders - available_keys_for_formatting
            raise TypeError(
                f"[Prompt Decorator Runtime Check Error for function '{self._original_func.__name__}']: "
                f"Not all placeholders in the template could be filled. Missing values for: {missing_keys}. "
                f"Available for formatting: {list(available_keys_for_formatting)}. "
                f"Original function args: {list(call_args.keys())}. "
                f"Returned by function (if dict): {list(returned_from_func.keys()) if isinstance(returned_from_func, dict) else '{}'}."
            )

        try:
            formatted_prompt_str = self._prompt_template.format(
                **args_for_formatting
            )
        except KeyError as e:
            raise TypeError(
                f"[Prompt Decorator Runtime Check Error during .format() for '{self._original_func.__name__}']: Missing key {e}. "
                f"Args used: {list(args_for_formatting.keys())}"
            ) from e

        return WrapperCallReturnObject(
            formatted_prompt=formatted_prompt_str, fn_name=self._original_func.__name__
        )


def prompt(prompt_template: str) -> Callable[[Callable[P, R_original]], PromptWrapper[P, R_original]]:
    """
    A decorator factory that creates a prompt-generating callable.
    The returned callable will format the prompt_template using arguments
    passed to it and values returned by the decorated function.
    Includes an experimental static check for typos in returned dictionary literals.
    """
    
    def decorator(func: Callable[P, R_original]) -> PromptWrapper[P, R_original]:
        # Calculate template_placeholders for AST check here, as it's needed at decoration time
        # This was previously done inside the Wrapper class, but AST check needs it earlier.
        # However, the AST check logic in the original code uses 'template_placeholders'
        # which was defined at the top of the 'prompt' function, so it should still be in scope here.
        # Let's ensure it's correctly scoped for the AST check.
        # The 'template_placeholders' for the AST check should be the one derived from 'prompt_template' argument of 'prompt'
        
        # Re-calculate or ensure 'template_placeholders' is accessible for AST check
        # This was defined at the top of the 'prompt' function in the original user code.
        # Let's make sure it's still accessible or recalculate it if necessary.
        # For the AST check, we need the placeholders from the `prompt_template` argument of the `prompt` function.
        local_template_placeholders = set(re.findall(r"\{(\w+)\}", prompt_template))

        # ---- Experimental AST-based static check (decoration time) ----
        try:
            source_code = inspect.getsource(func)
            dedented_source = textwrap.dedent(source_code)
            tree = ast.parse(dedented_source)

            func_def_node = None
            if (
                isinstance(tree, ast.Module)
                and tree.body
                and isinstance(tree.body[0], ast.FunctionDef)
            ):
                func_def_node = tree.body[0]
            elif isinstance(tree, ast.FunctionDef):
                func_def_node = tree

            if func_def_node and func_def_node.name == func.__name__:
                for node in ast.walk(func_def_node):
                    if isinstance(node, ast.Return) and isinstance(
                        node.value, ast.Dict
                    ):
                        literal_keys_in_return = []
                        for key_node in node.value.keys:
                            if isinstance(key_node, ast.Constant) and isinstance(
                                key_node.value, str
                            ):
                                literal_keys_in_return.append(key_node.value)
                            elif hasattr(ast, "Str") and isinstance(
                                key_node, ast.Str
                            ):  # Python < 3.8
                                literal_keys_in_return.append(key_node.s)

                        for returned_key in literal_keys_in_return:
                            if returned_key not in local_template_placeholders: # Use local_template_placeholders
                                close_matches = difflib.get_close_matches(
                                    returned_key,
                                    local_template_placeholders, # Use local_template_placeholders
                                    n=1,
                                    cutoff=0.75,
                                )
                                if close_matches:
                                    suggested_placeholder = close_matches[0]
                                    raise TypeError(
                                        f"[Prompt Decorator Static Check Error for function '{func.__name__}']:\n"
                                        f"  In a dictionary literal returned by the function, found key: '{returned_key}'.\n"
                                        f"  This key is NOT a defined placeholder in the prompt template.\n"
                                        f"  However, it's very similar to the placeholder: '{suggested_placeholder}'.\n"
                                        f"  This might be a typo. Please verify.\n"
                                        f"  (Note: This is a best-effort static check with limitations. The runtime check is definitive.)"
                                    )
        except (OSError, TypeError, IndentationError, SyntaxError) as e:
            warnings.warn(
                f"[Prompt Decorator Warning for function '{func.__name__}']: "
                f"Could not perform experimental static analysis of function body due to: {type(e).__name__}: {e}. "
                f"Relying solely on runtime checks.",
                UserWarning,
            )
        except Exception as e:
            warnings.warn(
                f"[Prompt Decorator Warning for function '{func.__name__}']: "
                f"An unexpected error occurred during experimental static analysis: {type(e).__name__}: {e}. "
                f"Relying solely on runtime checks.",
                UserWarning,
            )
        # ---- End of experimental AST-based static check ----

        return PromptWrapper(func, prompt_template)

    return decorator
