"""
A configurable mock inference endpoint (OpenAI chat-completions) that can be used for testing.

Idea: load some responses into a queue against a predictable sequence of requests for testing.
"""

from .endpoint import CompletionsEndpoint, Reply

__all__ = ["CompletionsEndpoint", "Reply"]
