"""
Test that agents are cleaned up when loop_on_socket exits.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentica_internal.session_manager_messages import CreateAgentRequest
from litestar.exceptions import WebSocketDisconnect
from litestar.status_codes import WS_1000_NORMAL_CLOSURE

from server_session_manager import InferenceProvider, ServerSessionManager


def create_session_manager():
    """Create a ServerSessionManager with minimal mock dependencies."""
    # Create a single provider from config
    providers = [
        InferenceProvider.from_config(
            {
                'endpoint': "https://openrouter.ai/api/v1/chat/completions",
                'token': 'test-token',
                'model_pattern': '*',
            }
        )
    ]

    return ServerSessionManager(
        log_poster=MagicMock(),
        providers=providers,
        user_id="test-user",
        sandbox_mode='no_sandbox',
    )


def create_mock_socket():
    """Create a mock WebSocket that disconnects immediately."""
    socket = MagicMock()
    socket.accept = AsyncMock()
    socket.close = AsyncMock()
    socket.send_bytes = AsyncMock()

    async def disconnect_immediately():
        raise WebSocketDisconnect(code=WS_1000_NORMAL_CLOSURE, detail="disconnected")

    socket.receive_bytes = disconnect_immediately
    return socket


@pytest.fixture
def suppress_ssm_logs():
    import logging

    logger = logging.getLogger('server_session_manager')
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(logging.ERROR)


@pytest.mark.asyncio
async def test_loop_on_socket_deregisters_session_and_cleans_up_agent(suppress_ssm_logs):
    """
    When loop_on_socket exits, the session should be deregistered
    and all agents in that session should be cleaned up.
    """
    sm = create_session_manager()
    cid = "test-session"

    # Register session
    sm.register_session(cid)

    # Mock dependencies for create_agent
    mock_inferencer = MagicMock()
    mock_inferencer.authenticate = AsyncMock()

    mock_model = MagicMock()
    mock_model.validate_openrouter_model = AsyncMock()

    mock_agent = MagicMock()
    mock_agent.iid = None
    mock_agent.aclose = MagicMock()

    with (
        patch(
            'server_session_manager.server_session_manager.ProviderModel.parse',
            return_value=mock_model,
        ),
        patch('server_session_manager.server_session_manager.Agent', return_value=mock_agent),
    ):
        # Create agent using the real method
        request = CreateAgentRequest(
            model="openai/gpt-4",
            doc=None,
            system=None,
            streaming=False,
            warp_globals_payload=b"",
        )
        uid = await sm.create_agent(request, cid)

        # Verify agent and session exist
        assert sm.has_agent(uid)
        assert sm.has_session(cid)

        # Run loop_on_socket - it will exit when socket disconnects
        socket = create_mock_socket()
        await sm.loop_on_socket(socket, cid=cid)

        # Verify session was deregistered and agent was cleaned up
        assert not sm.has_agent(uid)
        assert not sm.has_session(cid)
        mock_agent.aclose.assert_called_once()
