from .abstract import Converter
from .openai_completions import OpenAIChatCompletionConverter

__all__ = [
    'Convert',
    'Converter',
]

# === Namespacing converters ===


class Convert:
    OpenAIChatCompletions = OpenAIChatCompletionConverter
