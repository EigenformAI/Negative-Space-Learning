from typing import Dict, TypedDict, Any
from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    meta: NotRequired[Dict[str, Any]]
