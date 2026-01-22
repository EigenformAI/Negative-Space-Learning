from typing import List, Optional
from pydantic import BaseModel


class AppConfig(BaseModel):
    dev: bool
    model_name: str
    code_host_cache_path: str
    container_ids: List[str]
    main_container_idx: int

    # These 2 below are mutually exclusive
    dynamic_container: bool
    docker_compose_dir: str

    train_data_save_folder: str

    class PeftConfig(BaseModel):
        base_model_path: str
        checkpoint_path: str
        device: str = "auto"

    peft: Optional[PeftConfig] = None

    class SpecialEGCConfig(BaseModel):
        count: int
        max_retries: int

    special_egc: SpecialEGCConfig

    class StrategyListConfig(BaseModel):
        max_retries: int

    strategy_list: StrategyListConfig

    class StrategyCodeConfig(BaseModel):
        count: int
        max_retries: int

    strategy_code: StrategyCodeConfig
