from pydantic import BaseModel
from typing import Any, Dict


class ReturnType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ReturnType]
    returns: ReturnType


class PromptModel(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
