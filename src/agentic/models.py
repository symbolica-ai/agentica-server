from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from inference.endpoint import InferenceSystem

OPENROUTER_PREFIX = 'openrouter:'


@dataclass(kw_only=True)
class ProviderModel:
    provider: Literal['openai', 'anthropic'] | str
    model: str
    identifier: str
    endpoint_identifier: str

    async def validate_openrouter_model(self, endpoint: 'InferenceSystem') -> None:
        """Validate the endpoint identifier is reachable."""
        pass  # TODO: if we deem this functionality useful, we can re-implement it

    @classmethod
    def parse_openrouter(cls, pro_mod: str) -> 'ProviderModel':
        if '/' not in pro_mod:
            raise BadModel(f"Invalid OpenRouter model: `{pro_mod}`")

        provider, model = pro_mod.split('/', 1)
        return cls(
            provider=provider,
            model=model,
            identifier='openrouter:' + pro_mod,
            endpoint_identifier=pro_mod,
        )

    @classmethod
    def parse(cls, pro_mod: str) -> 'ProviderModel':
        if pro_mod.startswith(OPENROUTER_PREFIX):
            return cls.parse_openrouter(pro_mod[len(OPENROUTER_PREFIX) :])

        if ':' not in pro_mod or '/' in pro_mod:
            # just default to openrouter, even without the `openrouter:` prefix
            return cls.parse_openrouter(pro_mod)

        # Canonical model identifiers in provider/model format
        KNOWN_MODELS = {
            'openai': {
                'gpt-3.5-turbo': 'openai/gpt-3.5-turbo-instruct',
                'gpt-4o': 'openai/gpt-4o',
                'gpt-4.1': 'openai/gpt-4.1',
                'gpt-5': 'openai/gpt-5',
            },
            'anthropic': {
                'claude-sonnet-4': 'anthropic/claude-sonnet-4',
                'claude-opus-4.1': 'anthropic/claude-opus-4.1',
                'claude-sonnet-4.5': 'anthropic/claude-sonnet-4.5',
                'claude-opus-4.5': 'anthropic/claude-opus-4.5',
            },
        }

        provider, model = pro_mod.split(':', 1)

        if provider not in KNOWN_MODELS:
            raise BadModel(f"Invalid provider: `{provider}`")
        if model not in KNOWN_MODELS[provider]:
            raise BadModel(f"Invalid `{provider}` model: `{model}`")

        endpoint_identifier = KNOWN_MODELS[provider][model]

        return cls(
            provider=provider,
            model=model,
            identifier=pro_mod,
            endpoint_identifier=endpoint_identifier,
        )


class ValidationError(Exception):
    http_status_code: int = 400


class BadModel(ValidationError):
    """Bad model error."""

    def __init__(self, message: str):
        super().__init__(message)
