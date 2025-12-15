from enum import Enum

# from .openai_chat_completions import OpenAIChatCompletionInput
# from .vllm_openai_chat_completions import vLLMOpenAIChatCompletionsConfig


class API(Enum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    VLLM_OPENAI_CHAT_COMPLETIONS = "vllm_openai_chat_completions"


__all__ = [
    "API",
    # "OpenAIChatCompletionInput",
    # "vLLMOpenAIChatCompletionsConfig",
]
