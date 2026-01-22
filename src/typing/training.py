import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Type,
    TypeAlias,
    TypeVar,
    TypedDict,
    get_type_hints,
)
from typing_extensions import NotRequired
import json

from pydantic import BaseModel, Field

from src.typing.message import Message


class SpecialEnvironmentGetterCodeTrainData(TypedDict):
    # Global run id
    run_id: str
    version: str

    # Prompt to generate the code
    prompt: List[Message]
    # Raw response to generate the code
    raw_response: str

    # Current attempt for generating this code, indexed at 0, incremented at the start of the loop
    current_attempt: int
    # Max attempts for each code
    max_attempts: int

    # Result from `genner.extract_code`
    extract_result_if_ok: NotRequired[str]  # Code extracted
    extract_result_if_err: NotRequired[str]

    # Result from `validate_code_offline(extract_result_if_ok)`
    validation_result_if_err: NotRequired[str]

    # Result from `run_code_in_con(extract_result_if_ok)` after code validation
    run_result_if_ok: NotRequired[str]
    run_result_if_ok_hash: NotRequired[str]
    run_result_if_err: NotRequired[str]
    run_result_if_empty: NotRequired[str]


class StrategyListTrainData(TypedDict):
    # Global run id
    run_id: str
    version: str

    # Hashes of previous flow results
    special_env_info_hashes: List[str]

    # Prompt to generate the list
    prompt: List[Message]
    # Raw response of the list generation
    raw_response: str

    # Current attempt for generating this code, indexed at 0, incremented at the start of the loop
    current_attempt: int
    # Max attempts or retry
    max_attempts: int

    # Result from `genner.extract_list`
    extract_result_if_ok: NotRequired[List[str]]
    extract_result_if_ok_hash: NotRequired[str]
    extract_result_if_err: NotRequired[str]


class StrategyCodeTrainData(TypedDict):
    # Global run id
    run_id: str
    version: str

    # Hashes of previous flow results
    special_env_info_hashes: List[str]
    strategies_hash: str

    # Strategy to generate the code for
    strategy: str

    # Prompt to generate the code
    prompt: List[Message]
    # Raw response of the code generation
    raw_response: str

    # Current attempt for generating this code, indexed at 0, incremented at the start of the loop
    current_attempt: int
    # Max attempts or retry
    max_attempts: int

    # Result from `genner.extract_code`
    extract_result_if_err: NotRequired[str]
    extract_result_if_ok: NotRequired[str]  # This is also the code

    # Result from `genner.extract_code`
    validation_result_if_err: NotRequired[str]

    # Result from `run_code_in_con`
    run_result_if_ok: NotRequired[str]
    run_result_if_ok_hash: NotRequired[str]
    run_result_if_err: NotRequired[str]

    this_code_space_change_kb: NotRequired[float]


def convert_typed_dict_to_full_dict_generic(
    item: Mapping[str, Any], typed_dict_class: Type[Any]
) -> Dict[str, Any]:
    if not hasattr(typed_dict_class, "__annotations__") and not get_type_hints(
        typed_dict_class
    ):
        raise TypeError(
            f"'{typed_dict_class.__name__}' does not appear to be a TypedDict "
            "or lacks accessible type annotations."
        )

    # get_type_hints is generally more robust for getting annotations.
    all_keys = get_type_hints(typed_dict_class).keys()

    if not hasattr(item, "get"):
        raise TypeError(
            f"Input 'item' of type {type(item).__name__} is not dictionary-like and lacks a .get() method."
        )

    return {key: item.get(key) for key in all_keys}


TrainData: TypeAlias = (
    List[SpecialEnvironmentGetterCodeTrainData]
    | List[StrategyListTrainData]
    | List[StrategyCodeTrainData]
)

TrainDataType: TypeAlias = Literal[
    "special_environment_getter_code",
    "strategy_list",
    "strategy_code",
]


def save_train_data(
    train_data_type: TrainDataType, train_data: TrainData, folder_path: str | Path
) -> None:
    folder_path = Path(folder_path)

    for train_data_item in train_data:
        # Save with filename as {folder_path}/{version}_{run_id}_{train_data_type}_{YYYY-MM-DD-HH-MM-SS}.json
        if train_data_type == "special_environment_getter_code":
            train_data_item = convert_typed_dict_to_full_dict_generic(
                train_data_item, SpecialEnvironmentGetterCodeTrainData
            )
        elif train_data_type == "strategy_list":
            train_data_item = convert_typed_dict_to_full_dict_generic(
                train_data_item, StrategyListTrainData
            )
        elif train_data_type == "strategy_code":
            train_data_item = convert_typed_dict_to_full_dict_generic(
                train_data_item, StrategyCodeTrainData
            )

        cur_datetime = datetime.datetime.now()
        formatted_datetime = cur_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        run_id = train_data_item["run_id"]
        version = train_data_item["version"]

        filename = f"{version}_{run_id}_{train_data_type}_{formatted_datetime}.json"
        file_path = folder_path / filename

        # Create parent directories if they do not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(train_data_item, f, indent=4)
