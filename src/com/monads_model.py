from com.abstract import Do, HistoryMonad, Pure
from com.actions_model import *
from com.constraints import *
from com.context import Context
from com.deltas import *

__all__ = ['model_inference', 'insert_delta']

# Generation monad


def model_inference(
    stop_tokens: list[StopTokensType] = [],
    max_tokens: MaxTokensType | None = None,
    max_retries: int | None = None,
) -> HistoryMonad[GeneratedDelta]:
    """
    Query the inference endpoint w.r.t. the current context (history, model, global constraints)
    to retrieve a generated text delta `GeneratedDelta`.

    Additional constraints to generation may be specified.
    If no constraints are specified, all constraints will be inferred from the current context.

    Args:
        stop_tokens:
            Tokens that will cause the generation to stop. If more than one stop token is provided, the generation will stop when any of the tokens is generated.
        max_tokens:
            The maximum number of tokens to generate.
        max_retries:
            The maximum number of retries to make if the generation fails.
    """
    return Do(
        ModelInference(
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            max_retries=max_retries,
        ),
        Pure,
    )


def insert_delta(response: GeneratedDelta) -> HistoryMonad[Context]:
    """Update the monad context with a generated delta (ultimately from the result of a `gen()`)."""
    return Do(InsertDelta(response), Pure)
