import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient
from packaging.version import Version

from agentic.version_policy import SDK_VERSION_POLICIES
from application.main import SessionManager


@pytest.mark.asyncio
async def test_unsupported_version():
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
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
                    "json": False,
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
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
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
                    "json": False,
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
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
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
                    "json": False,
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
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        response = await client.post(
            "/agent/create",
            json={
                "doc": None,
                "system": None,
                "model": "openai:gpt-4o",
                "json": False,
                "streaming": False,
                "warp_globals_payload": "",
                "protocol": "invalid-format",
            },
            headers={"X-Client-Session-ID": "test-session"},
        )

        assert response.status_code in (400, 500)


@pytest.mark.asyncio
async def test_dev_versions():
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
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
                    "json": False,
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
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk in SDK_VERSION_POLICIES.keys():
            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "json": False,
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

    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
    app: Litestar = session_manager._app

    async with AsyncTestClient(app=app) as client:
        for sdk in SDK_VERSION_POLICIES.keys():
            response = await client.post(
                "/agent/create",
                json={
                    "doc": None,
                    "system": None,
                    "model": "openai:gpt-4o",
                    "json": False,
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
    session_manager = SessionManager(
        disable_otel=True,
        inference_endpoint="no-endpoint",
    )
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
                        "json": False,
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
