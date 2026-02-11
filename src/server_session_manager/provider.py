"""InferenceProvider - factory for creating InferenceSystem instances.

Each InferenceProvider owns an AsyncOpenAI client and creates InferenceSystem
instances for models matching its pattern.
"""

import fnmatch
import logging
from dataclasses import dataclass
from typing import Callable, TypedDict

from anthropic import AsyncAnthropic
from omegaconf import OmegaConf
from omegaconf.omegaconf import ListConfig
from openai import AsyncOpenAI

from agentic.models import ProviderModel
from application.http_client import get_client
from inference import ChatCompletionsSystem, InferenceSystem, MessagesSystem, ResponsesSystem
from messages import Notifier

from .server_session_manager import API, infer_api_from_endpoint

logger = logging.getLogger(__name__)


class InferenceProviderConfig(TypedDict):
    """Configuration for a single inference provider.

    Attributes:
        endpoint: Full URL ending with '/responses' or '/chat/completions'.
                  API type is inferred from the suffix.
        token: API key/token for authentication.
        model_pattern: fnmatch pattern for model matching (e.g., 'openai/*', '*').
    """

    endpoint: str
    token: str | None
    model_pattern: str | None  # default to '*' if not provided


def build_providers(
    *,
    inference_providers_path: str | None = None,
    legacy_endpoint: str | None = None,
    legacy_token: str | None = None,
) -> list["InferenceProvider"]:
    """Build list of InferenceProviders from configuration sources.

    Configuration is loaded in order of precedence (later overrides earlier):
    1. YAML config file (if inference_providers_path provided)
    2. Legacy CLI args (if no providers configured via above)

    Args:
        inference_providers_path: Path to YAML configuration file for inference providers.
        legacy_endpoint: Legacy --inference-endpoint value for backward compatibility.
        legacy_token: Legacy --inference-token value for backward compatibility.

    Returns:
        List of configured InferenceProvider instances.

    Raises:
        ValueError: If no providers are configured.
        omegaconf.errors.MissingMandatoryValue: If required fields are missing.
    """
    # Import here to avoid circular dependency
    from server_session_manager import InferenceProvider

    # 1. Load YAML config file if provided and resolve interpolations (e.g., ${oc.env:VAR})
    inference_providers = []
    if inference_providers_path:
        inference_providers = OmegaConf.load(inference_providers_path)
        OmegaConf.resolve(inference_providers)
        assert isinstance(inference_providers, ListConfig), (
            "Must specify inference providers as a list in a yaml/config config file."
        )

    # 2. Build provider list
    providers: list[InferenceProvider] = []

    for provider_cfg in inference_providers:
        providers.append(InferenceProvider.from_config(provider_cfg))

    # 3. Check the legacy CLI arguments
    if legacy_endpoint:
        if not providers:
            legacy_config = InferenceProviderConfig(
                endpoint=legacy_endpoint,
                token=legacy_token or "",
                model_pattern="*",
            )
            providers.append(InferenceProvider.from_config(legacy_config))
        else:
            raise ValueError(
                f"Cannot specify both --inference-providers and the (legacy) --inference-endpoint argument: {legacy_endpoint!r}, {providers!r}"
            )

    # Apply default fallback when no providers configured via any method
    if not providers:
        from application.defaults import DEFAULT_ENDPOINT_URL, DEFAULT_INFERENCE_TOKEN

        default_config = InferenceProviderConfig(
            endpoint=DEFAULT_ENDPOINT_URL,
            token=DEFAULT_INFERENCE_TOKEN,
            model_pattern="*",
        )
        providers.append(InferenceProvider.from_config(default_config))

    return providers


class NoMatchingProviderError(Exception):
    """Raised when no provider matches the requested model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"No provider configured for model: {model_id}")


@dataclass
class InferenceProvider:
    """Factory for creating InferenceSystem instances for matching models.

    Attributes:
        client: AsyncOpenAI or AsyncAnthropic client configured for this provider.
        model_pattern: fnmatch pattern for model matching.
        inference_system_cls: ResponsesSystem, ChatCompletionsSystem, or MessagesSystem.
    """

    client: AsyncOpenAI | AsyncAnthropic
    model_pattern: str
    inference_system_cls: type[ResponsesSystem] | type[ChatCompletionsSystem] | type[MessagesSystem]

    def matches(self, model_id: str) -> bool:
        """Check if this provider handles the given model_id."""
        return fnmatch.fnmatch(model_id, self.model_pattern)

    def create_inference_system(
        self,
        model_id: str,
        fresh_id: Callable[[], str],
        notifier: Notifier,
    ) -> InferenceSystem | None:
        """Create an InferenceSystem if this provider handles model_id.

        Args:
            model_id: The model identifier (e.g., 'openai/gpt-5').
            fresh_id: Callable that generates unique IDs.
            notifier: Notifier for sending events.

        Returns:
            InferenceSystem instance if this provider matches, None otherwise.
        """
        if not self.matches(model_id):
            return None

        model = ProviderModel.parse(model_id)
        return self.inference_system_cls(
            client=self.client,
            fresh_id=fresh_id,
            notifier=notifier,
            model=model,
        )

    @classmethod
    def from_config(cls, config: InferenceProviderConfig) -> "InferenceProvider":
        """Create an InferenceProvider from configuration.

        Args:
            config: InferenceProviderConfig with endpoint, token, pattern.

        Returns:
            Configured InferenceProvider with client and inference system class.

        Raises:
            ValueError: If endpoint URL doesn't end with recognized API path.
        """
        api_key = config.get('token', None) or ""
        if api_key == "":
            logger.warning("Inference provider specified without a token")

        base_url, api_type = infer_api_from_endpoint(config['endpoint'])

        if api_type == API.MESSAGES:
            # Anthropic client requires api_key to be set, use dummy for mock servers
            anthropic_key = api_key if api_key else "dummy-key-for-testing"
            client: AsyncOpenAI | AsyncAnthropic = AsyncAnthropic(
                api_key=anthropic_key,
                base_url=base_url,
                http_client=get_client(),
            )
        else:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=get_client(),
            )

        inference_system_cls: (
            type[ResponsesSystem] | type[ChatCompletionsSystem] | type[MessagesSystem]
        ) = {
            API.RESPONSES: ResponsesSystem,
            API.CHAT_COMPLETIONS: ChatCompletionsSystem,
            API.MESSAGES: MessagesSystem,
        }[api_type]

        return cls(
            client=client,
            model_pattern=config.get('model_pattern', None) or "*",
            inference_system_cls=inference_system_cls,
        )
