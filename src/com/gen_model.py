import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from agentica_internal.session_manager_messages import CacheTTL, ReasoningEffort
from openai.types.responses import ResponseUsage

if TYPE_CHECKING:
    from agentica_internal.internal_errors import GenerationError


__all__ = [
    'InferenceConfig',
    'GenModelErrorHandling',
    'GenCacheKey',
]


type SendGenErrFn = Callable[[GenerationError], Awaitable[None]]

_default_cache_key = lambda: GenCacheKey()
_default_error_handling = lambda: GenModelErrorHandling()


# user messages have optional userNAME associated. Some providers support this, some don't.
# e.g. ('user', 'John') means "A user called John" to openai models.
type Role = Literal['assistant', 'system'] | tuple[Literal['user'], str | None]
type FinishReason = Literal['done', 'tool', 'max_tokens', 'content_filter']


@dataclass
class Generation:
    output_text: str  # The assistant text WITHOUT the code
    code: str | None
    usage: ResponseUsage
    reasoning: str | None = None
    # Number of extra code blocks that were ignored (only relevant for markdown extraction).
    # Will be removed when Chat Completions support is dropped.
    extra_code_blocks: int = 0
    # True if code came from a tool call (Responses API), False if from markdown extraction
    code_from_tool: bool = False

    @property
    def content(self) -> str:
        """Alias for output_text, for backwards compatibility with code expecting GeneratedDelta."""
        return self.output_text


@dataclass
class Delta:
    content: str
    type: Literal['reasoning', 'output_text', 'code', 'task']

    @property
    def role(self) -> Literal['user', 'agent']:
        return 'user' if self.type == 'task' else "agent"


@dataclass
class InferenceConfig:
    """
    Model inference / generation context. Should rename to ModelContext or something.
    """

    iid: str
    model: str
    max_rounds: int | None  # Maximum number of rounds of inference (unlimited if None)
    max_invocation_tokens: int | None  # Total tokens allowed for the invocation (unlimited if None)
    max_inference_tokens: int | None  # Max tokens for a single inference (unlimited if None)
    max_completion_tokens: int | None  # Subtracted from each round of inference (unlimited if None)
    # `max_tokens` becomes `min(max_inference_tokens, max_completion_tokens)`
    send_gen_err: SendGenErrFn
    streaming: bool = False
    reasoning_effort: ReasoningEffort | None = None
    cache_ttl: CacheTTL | None = None
    inference_rounds_count: int = 0
    cache_key: 'GenCacheKey' = field(default_factory=_default_cache_key)
    error_handling: 'GenModelErrorHandling' = field(default_factory=_default_error_handling)

    def max_tokens(self) -> int | None:
        if self.max_inference_tokens is None:
            # if no max per round of inference, use the max per completion
            return self.max_completion_tokens
        if self.max_completion_tokens is None:
            # if no max per completion, use the max per round of inference
            return self.max_inference_tokens
        # if both are set, use the minimum of the two for limiting the inference round
        return min(self.max_inference_tokens, self.max_completion_tokens)

    def finish_invocation(self) -> None:
        self.max_completion_tokens = self.max_invocation_tokens
        self.inference_rounds_count = 0

    def spend_tokens(self, tokens: int) -> None:
        """Subtract the given number of tokens from our limit."""
        mct = self.max_completion_tokens
        if mct is None:
            return
        mct = max(mct - tokens, 0)
        self.max_completion_tokens = mct

    def use_cache_key(self, condition: Callable[['GenCacheKey'], bool]) -> str:
        """
        Use the last cache key if it satisfies the condition, otherwise create a new cache key.
        """
        # Note cache keys are created with a default first_request_time of 0 so time dependent
        # conditions are always true on the first request
        if not condition(self.cache_key):
            self.cache_key = GenCacheKey(first_request_time=time.time())
        self.cache_key.num_requests += 1
        return self.cache_key.key

    def __str__(self) -> str:
        return f"InferenceConfig(model={self.model}, max_rounds={self.max_rounds})"


@dataclass
class GenModelErrorHandling:
    """
    Error handling configuration.
    """

    max_retries: int = 1
    rate_limit_delay: float = 1
    rate_limit_exponential_base: float = 2
    rate_limit_jitter: bool = True
    read_timeout: int | None = None


@dataclass
class GenCacheKey:
    """
    A cache key for a generation.
    """

    key: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_request_time: float = 0
    num_requests: int = 0
