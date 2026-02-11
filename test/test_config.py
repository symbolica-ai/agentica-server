"""Tests for application.config module - provider configuration building."""

import tempfile
from pathlib import Path

import pytest

from server_session_manager.provider import build_providers


class TestBuildProviders:
    """Tests for build_providers function."""

    def test_legacy_args_chat_completions(self):
        """Legacy CLI args with chat/completions endpoint."""
        providers = build_providers(
            legacy_endpoint="https://openrouter.ai/api/v1/chat/completions",
            legacy_token="test-token",
        )

        assert len(providers) == 1
        provider = providers[0]
        assert str(provider.client.base_url) == "https://openrouter.ai/api/v1/"
        assert provider.client.api_key == "test-token"
        assert provider.model_pattern == "*"
        assert provider.inference_system_cls.__name__ == "ChatCompletionsSystem"

    def test_legacy_args_responses(self):
        """Legacy CLI args with responses endpoint."""
        providers = build_providers(
            legacy_endpoint="https://api.openai.com/v1/responses",
            legacy_token="sk-test",
        )

        assert len(providers) == 1
        provider = providers[0]
        assert str(provider.client.base_url) == "https://api.openai.com/v1/"
        assert provider.client.api_key == "sk-test"
        assert provider.model_pattern == "*"
        assert provider.inference_system_cls.__name__ == "ResponsesSystem"

    def test_yaml_config_file(self):
        """Load providers from YAML config file."""
        yaml_content = """
- endpoint: https://api.openai.com/v1/responses
  token: sk-yaml
  model_pattern: "openai/*"
- endpoint: https://openrouter.ai/api/v1/chat/completions
  token: or-yaml
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            inference_providers_path = f.name

        try:
            providers = build_providers(inference_providers_path=inference_providers_path)

            assert len(providers) == 2

            # First provider
            assert str(providers[0].client.base_url) == "https://api.openai.com/v1/"
            assert providers[0].client.api_key == "sk-yaml"
            assert providers[0].model_pattern == "openai/*"
            assert providers[0].inference_system_cls.__name__ == "ResponsesSystem"

            # Second provider (default model_pattern)
            assert str(providers[1].client.base_url) == "https://openrouter.ai/api/v1/"
            assert providers[1].client.api_key == "or-yaml"
            assert providers[1].model_pattern == "*"
            assert providers[1].inference_system_cls.__name__ == "ChatCompletionsSystem"
        finally:
            Path(inference_providers_path).unlink()

    def test_no_providers_uses_defaults(self):
        """Falls back to default endpoint when no providers configured."""
        from application.defaults import DEFAULT_ENDPOINT_URL

        providers = build_providers()

        assert len(providers) == 1
        provider = providers[0]
        # Default endpoint is openrouter chat/completions
        assert (
            "openrouter.ai" in str(provider.client.base_url)
            or "openrouter" in DEFAULT_ENDPOINT_URL.lower()
        )
        assert provider.model_pattern == "*"

    def test_yaml_config_file_platform_urls(self):
        """Load providers from YAML config file."""
        yaml_content = """
- endpoint: https://inference-service-prod.fly.dev/v1/responses
  token: sk-yaml
  model_pattern: "openai/*"
- endpoint: https://inference-service-prod.fly.dev/v1/chat/completions
  token: or-yaml
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            inference_providers_path = f.name

        try:
            providers = build_providers(inference_providers_path=inference_providers_path)

            assert len(providers) == 2

            # First provider
            assert str(providers[0].client.base_url) == "https://inference-service-prod.fly.dev/v1/"
            assert providers[0].client.api_key == "sk-yaml"
            assert providers[0].model_pattern == "openai/*"
            assert providers[0].inference_system_cls.__name__ == "ResponsesSystem"

            # Second provider (default model_pattern)
            assert str(providers[1].client.base_url) == "https://inference-service-prod.fly.dev/v1/"
            assert providers[1].client.api_key == "or-yaml"
            assert providers[1].model_pattern == "*"
            assert providers[1].inference_system_cls.__name__ == "ChatCompletionsSystem"
        finally:
            Path(inference_providers_path).unlink()

    def test_providers_and_legacy_endpoint_raises(self):
        """Raises ValueError when both providers and legacy endpoint are specified."""
        yaml_content = """
- endpoint: https://api.openai.com/v1/responses
  token: sk-yaml
  model_pattern: "openai/*"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            inference_providers_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot specify both"):
                build_providers(
                    inference_providers_path=inference_providers_path,
                    legacy_endpoint="https://openrouter.ai/api/v1/chat/completions",
                )
        finally:
            Path(inference_providers_path).unlink()
