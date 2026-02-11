import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient
from packaging.version import Version

from agentic.version_policy import SDK_VERSION_POLICIES, VersionStatus, check_sdk_version
from application.main import SessionManager
from server_session_manager import InferenceProvider


def create_test_session_manager():
    """Helper to create SessionManager with test provider configuration."""
    providers = [
        InferenceProvider.from_config(
            {
                'endpoint': "https://openrouter.ai/api/v1/chat/completions",
                'token': '',
                'model_pattern': '*',
            }
        )
    ]
    return SessionManager(disable_otel=True, providers=providers)


class TestDateBasedVersionNormalization:
    """
    Test that packaging.version.Version correctly handles date-based versions
    with leading zeros, as used by the TypeScript SDK.

    TypeScript SDK emits versions like: 2026.01.20-dev0+28e2fc0
    Python SDK / server emits versions like: 2026.1.18.dev3

    These must compare correctly despite format differences.
    """

    @pytest.mark.parametrize(
        "version_str,expected_normalized",
        [
            ("2026.01.17", "2026.1.17"),
            ("2026.1.17", "2026.1.17"),
            ("2026.01.17.dev0+abc123", "2026.1.17.dev0+abc123"),
            ("2026.1.17.dev0+abc123", "2026.1.17.dev0+abc123"),
            ("2026.01.17-dev0+abc123", "2026.1.17.dev0+abc123"),
            ("2026.1.17-dev0+abc123", "2026.1.17.dev0+abc123"),
            ("2026.01.20-dev0+28e2fc0", "2026.1.20.dev0+28e2fc0"),
        ],
    )
    def test_version_parsing_normalizes_leading_zeros(
        self, version_str: str, expected_normalized: str
    ):
        """Verify that packaging.version normalizes leading zeros in date components."""
        parsed = Version(version_str)
        assert str(parsed) == expected_normalized

    def test_versions_with_and_without_leading_zeros_are_equal(self):
        """Versions with and without leading zeros should compare as equal."""
        v1 = Version("2026.01.17.dev0")
        v2 = Version("2026.1.17.dev0")
        assert v1 == v2

    def test_typescript_sdk_version_newer_than_server_minimum(self):
        """
        TypeScript SDK version 2026.01.20-dev0+28e2fc0 should be correctly
        identified as newer than server minimum 2026.1.18.dev3.
        """
        typescript_sdk_version = Version("2026.01.20-dev0+28e2fc0")
        server_minimum = Version("2026.1.18.dev3")
        assert typescript_sdk_version > server_minimum

    @pytest.mark.parametrize("sdk", ["python", "typescript"])
    def test_check_sdk_version_accepts_date_based_versions(self, sdk: str):
        """Date-based versions from both SDKs should be accepted by check_sdk_version."""
        # Use a version far in the future to ensure it passes
        future_version = "2099.01.01-dev0+abc123"
        status = check_sdk_version(sdk, future_version)
        assert status == VersionStatus.OK


@pytest.fixture
def suppress_smapp_logs():
    import logging

    logger = logging.getLogger('session_manager_application')
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(logging.ERROR)


@pytest.mark.asyncio
async def test_unsupported_version(suppress_smapp_logs):
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    unsupported_version = "0.1.99"

    async with AsyncTestClient(app=app) as client:
        for sdk, policy in SDK_VERSION_POLICIES.items():
            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/{unsupported_version}",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 426, f"Failed for SDK: {sdk}"
            assert "SDK VERSION NOT SUPPORTED" in response.text
            assert unsupported_version in response.text
            assert str(policy.min_supported.public) in response.text


@pytest.mark.asyncio
async def test_deprecated_version():
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk, policy in SDK_VERSION_POLICIES.items():
            min_supported = policy.min_supported
            min_recommended = policy.min_recommended

            if min_supported >= min_recommended:
                pytest.skip(f"No deprecated range for {sdk}: min_supported >= min_recommended")

            # Use min_supported as the deprecated version
            deprecated_version = min_supported

            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/{deprecated_version.public}",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 201, f"Failed for SDK: {sdk}"
            assert response.headers.get("X-SDK-Warning") == "deprecated"
            upgrade_message = response.headers.get("X-SDK-Upgrade-Message")
            assert upgrade_message is not None, f"Missing upgrade message for SDK: {sdk}"
            assert "SDK update recommended" in upgrade_message, f"Wrong message for SDK: {sdk}"
            assert str(deprecated_version.public) in upgrade_message, (
                f"Version not in message for SDK: {sdk}"
            )


@pytest.mark.asyncio
async def test_current_version():
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk, policy in SDK_VERSION_POLICIES.items():
            min_recommended = policy.min_recommended

            # Use a version higher than min_recommended
            current_version = Version(
                f"{min_recommended.major}.{min_recommended.minor}.{min_recommended.micro + 1}"
            )

            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/{current_version.public}",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 201, f"Failed for SDK: {sdk}"
            assert "X-SDK-Warning" not in response.headers
            assert "X-SDK-Upgrade-Message" not in response.headers


@pytest.mark.asyncio
async def test_invalid_protocol_format():
    from os import getenv

    if getenv('LOCAL_TESTING'):
        pytest.skip("un-suppressible error logs pollute ./run_session_manager_tests.sh")

    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        response = await client.post(
            "/agent/create",
            json={
                "doc": None,
                "system": None,
                "model": "openai:gpt-4o",
                "streaming": False,
                "warp_globals_payload": "",
                "protocol": "invalid-format",
            },
            headers={"X-Client-Session-ID": "test-session"},
        )

        assert response.status_code in (400, 500)


@pytest.mark.asyncio
async def test_dev_versions():
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk, policy in SDK_VERSION_POLICIES.items():
            min_recommended = policy.min_recommended

            dev_version = (
                f"{min_recommended.major}.{min_recommended.minor}.{min_recommended.micro + 1}"
                f".dev215+ge77ba7d9c.d20251104"
            )

            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/{dev_version}",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 201, f"Failed for SDK: {sdk}"
            assert "X-SDK-Warning" not in response.headers


@pytest.mark.asyncio
async def test_local_development_version():
    """Test that 0.0.0-dev is allowed in local mode (default test environment)."""
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk in SDK_VERSION_POLICIES.keys():
            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/0.0.0-dev",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 201, f"Failed for SDK: {sdk}"
            assert "X-SDK-Warning" not in response.headers


@pytest.mark.asyncio
async def test_local_development_version_blocked_in_production(monkeypatch):
    """Test that 0.0.0-dev is blocked when ORGANIZATION_ID is set (production mode)."""
    # Simulate production environment by setting ORGANIZATION_ID
    monkeypatch.setenv("ORGANIZATION_ID", "prod-org-123")

    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk in SDK_VERSION_POLICIES.keys():
            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "streaming": False,
                    "warp_globals_payload": "",
                    "protocol": f"{sdk}/0.0.0-dev",
                },
                headers={"X-Client-Session-ID": "test-session"},
            )

            assert response.status_code == 426, (
                f"0.0.0-dev should be blocked in production for SDK: {sdk}"
            )
            assert "SDK VERSION NOT SUPPORTED" in response.text


@pytest.mark.asyncio
async def test_malformed_version():
    session_manager = create_test_session_manager()
    app: Litestar = session_manager._app

    # These versions are truly malformed per PEP 440
    # Note: packaging.version is lenient and accepts things like "1.2.3.4.5" and "v1.2.3"
    malformed_versions = [
        "not-a-version",
        "abc.def.ghi",
        "",
        "...",
        "x.y.z",
        "1.2.3-not-valid",
    ]

    async with AsyncTestClient(app=app) as client:
        for sdk in SDK_VERSION_POLICIES.keys():
            for malformed_version in malformed_versions:
                response = await client.post(
                    "/agent/create",
                    json={
                        "doc": None,
                        "system": None,
                        "model": "openai:gpt-4o",
                        "streaming": False,
                        "warp_globals_payload": "",
                        "protocol": f"{sdk}/{malformed_version}",
                    },
                    headers={"X-Client-Session-ID": "test-session"},
                )

                assert response.status_code == 426, (
                    f"Failed for SDK: {sdk}, version: {malformed_version}"
                )
                assert "SDK VERSION NOT SUPPORTED" in response.text
