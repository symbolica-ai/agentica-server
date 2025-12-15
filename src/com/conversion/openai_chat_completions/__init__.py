from typing import TYPE_CHECKING

from com.deltas import Delta

from .anthropic import convert_from_deltas as convert_from_deltas_anthropic
from .openai import convert_from_deltas as convert_from_deltas_openai
from .utils import convert_from_constraints, convert_to_delta

__all__ = [
    'convert_from_deltas',
    'convert_to_delta',
    'convert_from_constraints',
]

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


def convert_from_deltas(
    deltas: list[Delta], model_provider: str
) -> list['ChatCompletionMessageParam']:
    if model_provider == 'openai':
        return convert_from_deltas_openai(deltas)
    elif model_provider == 'anthropic':
        return convert_from_deltas_anthropic(deltas)
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")
