"""
Unit tests for multiplexer handling of malformed messages.

Tests that the multiplexer correctly sends error responses when receiving
malformed or invalid multiplex messages.

Test cases:
- test_invoke_with_unknown_uid_sends_error: Invoke with unknown uid → MalformedInvokeMessageError
- test_invoke_with_empty_uid_sends_error: Invoke with empty uid → MalformedInvokeMessageError
- test_cancel_with_unknown_iid_sends_error: Cancel with unknown iid → NotRunningError
- test_data_with_unknown_iid_sends_error: Data with unknown iid → NotRunningError
- test_multiple_malformed_messages: Multiple malformed messages in sequence
- test_cancel_with_wrong_uid_sends_error: Cancel with wrong uid → NotRunningError
- test_data_with_wrong_uid_sends_error: Data with wrong uid → NotRunningError
- test_interleaved_valid_and_invalid_invokes: Mix of valid and invalid invokes
- test_interleaved_valid_invoke_then_invalid_cancel_data: Valid invoke followed by invalid cancel/data
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agentica_internal.multiplex_protocol import (
    MultiplexCancelMessage,
    MultiplexDataMessage,
    MultiplexErrorMessage,
    MultiplexInvokeMessage,
    MultiplexServerMessage,
    multiplex_to_json,
)
from litestar.exceptions import WebSocketDisconnect
from litestar.status_codes import WS_1000_NORMAL_CLOSURE

from server_session_manager.multiplexer import Multiplexer, ServerSessionContext


def make_recv_bytes(messages: list):
    """
    Create a recv_bytes function that yields message bytes, then disconnects.
    """
    message_bytes = [multiplex_to_json(msg) for msg in messages]
    index = 0

    async def recv_bytes() -> bytes:
        nonlocal index
        if index < len(message_bytes):
            result = message_bytes[index]
            index += 1
            return result
        # Signal end of messages with normal closure
        raise WebSocketDisconnect(code=WS_1000_NORMAL_CLOSURE, detail="test complete")

    return recv_bytes


@pytest.fixture
def captured_messages():
    """List to capture messages sent through the notifier."""
    return []


@pytest.fixture
def suppress_multiplexer_logs():
    import logging

    logger = logging.getLogger('server_session_manager.multiplexer')
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(logging.ERROR)


@pytest.fixture
def mock_server_session_ctx(captured_messages):
    """Create a mock ServerSessionContext with empty agents."""
    ctx = MagicMock(spec=ServerSessionContext)
    ctx.agents = {}  # No agents registered

    # Create a notifier that captures messages
    async def capture_send(msg: MultiplexServerMessage) -> None:
        captured_messages.append(msg)

    ctx.notifier = MagicMock()
    ctx.notifier.send_mx_message = capture_send

    ctx.log_poster = MagicMock()
    ctx.logs = MagicMock()
    ctx.tracer = None
    ctx.parent_context = None
    ctx.register_invocation_span = MagicMock()
    ctx.get_invocation_span = MagicMock()
    ctx.user_id = None
    ctx.uid_to_cid = {}
    return ctx


def create_multiplexer(
    mock_server_session_ctx, messages: list, captured_messages: list | None = None
) -> Multiplexer:
    """Create a Multiplexer that will process the given messages."""

    async def mock_send_bytes(data: bytes) -> None:
        pass

    async def mock_transport_enqueue(msg: MultiplexServerMessage) -> None:
        # Capture messages sent via transport (used by valid agent responses)
        if captured_messages is not None:
            captured_messages.append(msg)

    return Multiplexer(
        fresh_id=lambda: "test-iid",
        send_bytes=mock_send_bytes,
        recv_bytes=make_recv_bytes(messages),
        transport_enqueue=mock_transport_enqueue,
        server_session_ctx=mock_server_session_ctx,
        create_invocation=AsyncMock(return_value=True),
        destroy_invocation=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_invoke_with_unknown_uid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    When receiving an invoke with unknown uid, multiplexer sends MalformedInvokeMessageError.
    """
    messages = [
        MultiplexInvokeMessage(
            match_id="match-123",
            uid="unknown-agent-uid",
            warp_locals_payload=b"",
            streaming=False,
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "match-123"
    assert "malformedinvokemessageerror" in error_msg.error_name.lower()


@pytest.mark.asyncio
async def test_invoke_with_empty_uid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    When receiving an invoke with empty uid, multiplexer sends MalformedInvokeMessageError.
    """
    messages = [
        MultiplexInvokeMessage(
            match_id="match-empty",
            uid="",
            warp_locals_payload=b"",
            streaming=False,
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "match-empty"
    assert "malformedinvokemessageerror" in error_msg.error_name.lower()


@pytest.mark.asyncio
async def test_cancel_with_unknown_iid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    When receiving a cancel with unknown iid, multiplexer sends NotRunningError.
    """
    messages = [
        MultiplexCancelMessage(
            uid="some-uid",
            iid="unknown-iid",
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "unknown-iid"
    assert "notrunningerror" in error_msg.error_name.lower()


@pytest.mark.asyncio
async def test_data_with_unknown_iid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    When receiving data with unknown iid, multiplexer sends NotRunningError.
    """
    messages = [
        MultiplexDataMessage(
            uid="some-uid",
            iid="unknown-data-iid",
            data=b"test data",
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "unknown-data-iid"
    assert "notrunningerror" in error_msg.error_name.lower()


@pytest.mark.asyncio
async def test_multiple_malformed_messages(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    Multiple malformed messages each receive their own error response.
    """
    messages = [
        MultiplexInvokeMessage(
            match_id="m1", uid="bad-uid-1", warp_locals_payload=b"", streaming=False
        ),
        MultiplexInvokeMessage(
            match_id="m2", uid="bad-uid-2", warp_locals_payload=b"", streaming=False
        ),
        MultiplexCancelMessage(uid="x", iid="bad-iid-1"),
        MultiplexDataMessage(uid="x", iid="bad-iid-2", data=b""),
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 4

    # First two should be MalformedInvokeMessageError
    assert "malformedinvokemessageerror" in captured_messages[0].error_name.lower()
    assert captured_messages[0].iid == "m1"
    assert "malformedinvokemessageerror" in captured_messages[1].error_name.lower()
    assert captured_messages[1].iid == "m2"

    # Last two should be NotRunningError
    assert "notrunningerror" in captured_messages[2].error_name.lower()
    assert captured_messages[2].iid == "bad-iid-1"
    assert "notrunningerror" in captured_messages[3].error_name.lower()
    assert captured_messages[3].iid == "bad-iid-2"


@pytest.mark.asyncio
async def test_cancel_with_wrong_uid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    Cancel with wrong uid (iid doesn't exist) returns NotRunningError.
    """
    messages = [
        MultiplexCancelMessage(
            uid="wrong-agent-uid",
            iid="nonexistent-iid",
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "nonexistent-iid"
    assert "notrunningerror" in error_msg.error_name.lower()


@pytest.mark.asyncio
async def test_data_with_wrong_uid_sends_error(
    mock_server_session_ctx, captured_messages, suppress_multiplexer_logs
):
    """
    Data with wrong uid (iid doesn't exist) returns NotRunningError.
    """
    messages = [
        MultiplexDataMessage(
            uid="wrong-agent-uid",
            iid="nonexistent-data-iid",
            data=b"payload",
        )
    ]

    mux = create_multiplexer(mock_server_session_ctx, messages)
    await mux.run()

    assert len(captured_messages) == 1
    error_msg = captured_messages[0]
    assert isinstance(error_msg, MultiplexErrorMessage)
    assert error_msg.iid == "nonexistent-data-iid"
    assert "notrunningerror" in error_msg.error_name.lower()


@pytest.fixture
def mock_server_session_ctx_with_valid_agent(captured_messages):
    """Create a mock ServerSessionContext with one valid agent registered."""
    ctx = MagicMock(spec=ServerSessionContext)

    # Create a mock Agent
    mock_agentic = MagicMock()
    mock_agentic.model = MagicMock()
    mock_agentic.model.provider = "test-provider"
    mock_agentic.model.identifier = "test-model"
    mock_agentic.session_id = "test-session"
    mock_agentic.session_manager_id = "test-sm"
    mock_agentic.run = AsyncMock()  # Mock the run method to do nothing
    mock_agentic.cancel = MagicMock()

    # Register the valid agent
    ctx.agents = {"valid-agent-uid": mock_agentic}

    # Create a notifier that captures messages (used for errors)
    async def capture_send(msg: MultiplexServerMessage) -> None:
        captured_messages.append(msg)

    ctx.notifier = MagicMock()
    ctx.notifier.send_mx_message = capture_send

    ctx.log_poster = MagicMock()
    ctx.logs = MagicMock()
    ctx.tracer = None
    ctx.parent_context = None
    ctx.register_invocation_span = MagicMock()
    ctx.get_invocation_span = MagicMock()
    ctx.user_id = None
    ctx.uid_to_cid = {}
    return ctx


def create_multiplexer_with_valid_agent(
    mock_server_session_ctx, messages: list, captured_messages: list
) -> Multiplexer:
    """Create a Multiplexer with a valid agent - captures both transport and error messages."""

    async def mock_send_bytes(data: bytes) -> None:
        pass

    async def mock_transport_enqueue(msg: MultiplexServerMessage) -> None:
        # Valid agent responses go through transport_enqueue
        captured_messages.append(msg)

    return Multiplexer(
        fresh_id=lambda: "test-iid",
        send_bytes=mock_send_bytes,
        recv_bytes=make_recv_bytes(messages),
        transport_enqueue=mock_transport_enqueue,
        server_session_ctx=mock_server_session_ctx,
        create_invocation=AsyncMock(return_value=True),
        destroy_invocation=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_interleaved_valid_and_invalid_invokes(
    mock_server_session_ctx_with_valid_agent, captured_messages, suppress_multiplexer_logs
):
    """
    Interleaved valid and invalid invoke messages.
    Valid invokes get MultiplexNewIIDResponse, invalid get MalformedInvokeMessageError.
    """
    from agentica_internal.multiplex_protocol import MultiplexNewIIDResponse

    messages = [
        # Valid invoke to registered agent
        MultiplexInvokeMessage(
            match_id="valid-1", uid="valid-agent-uid", warp_locals_payload=b"", streaming=False
        ),
        # Invalid invoke to unknown agent
        MultiplexInvokeMessage(
            match_id="invalid-1", uid="unknown-uid", warp_locals_payload=b"", streaming=False
        ),
        # Another valid invoke
        MultiplexInvokeMessage(
            match_id="valid-2", uid="valid-agent-uid", warp_locals_payload=b"", streaming=False
        ),
        # Invalid invoke with empty uid
        MultiplexInvokeMessage(
            match_id="invalid-2", uid="", warp_locals_payload=b"", streaming=False
        ),
        # Another valid invoke
        MultiplexInvokeMessage(
            match_id="valid-3", uid="valid-agent-uid", warp_locals_payload=b"", streaming=False
        ),
    ]

    mux = create_multiplexer_with_valid_agent(
        mock_server_session_ctx_with_valid_agent, messages, captured_messages
    )
    await mux.run()

    # Filter to just NewIIDResponse and MalformedInvokeMessageError
    # (ignore InternalServerError from notifier.on_exception - those are tested elsewhere)
    responses = [
        m
        for m in captured_messages
        if isinstance(m, MultiplexNewIIDResponse)
        or (
            isinstance(m, MultiplexErrorMessage)
            and "malformedinvokemessageerror" in str(m.error_name).lower()
        )
    ]

    # Should have 5 responses: 3 valid (NewIIDResponse) + 2 invalid (MalformedInvokeMessageError)
    assert len(responses) == 5

    # valid-1: should get MultiplexNewIIDResponse
    assert isinstance(responses[0], MultiplexNewIIDResponse)
    assert responses[0].match_id == "valid-1"

    # invalid-1: should get MalformedInvokeMessageError
    assert isinstance(responses[1], MultiplexErrorMessage)
    assert responses[1].iid == "invalid-1"
    assert "malformedinvokemessageerror" in responses[1].error_name.lower()

    # valid-2: should get MultiplexNewIIDResponse
    assert isinstance(responses[2], MultiplexNewIIDResponse)
    assert responses[2].match_id == "valid-2"

    # invalid-2: should get MalformedInvokeMessageError
    assert isinstance(responses[3], MultiplexErrorMessage)
    assert responses[3].iid == "invalid-2"
    assert "malformedinvokemessageerror" in responses[3].error_name.lower()

    # valid-3: should get MultiplexNewIIDResponse
    assert isinstance(responses[4], MultiplexNewIIDResponse)
    assert responses[4].match_id == "valid-3"


@pytest.mark.asyncio
async def test_interleaved_valid_invoke_then_invalid_cancel_data(
    mock_server_session_ctx_with_valid_agent, captured_messages
):
    """
    Valid invoke followed by invalid cancel/data for wrong iids.
    Shows that after a valid invoke, invalid cancel/data still get proper errors.
    """
    from agentica_internal.multiplex_protocol import MultiplexNewIIDResponse

    messages = [
        # Valid invoke - will create iid "test-iid" (from fresh_id mock)
        MultiplexInvokeMessage(
            match_id="valid-invoke", uid="valid-agent-uid", warp_locals_payload=b"", streaming=False
        ),
        # Invalid cancel - wrong iid
        MultiplexCancelMessage(uid="valid-agent-uid", iid="wrong-iid"),
        # Invalid data - wrong iid
        MultiplexDataMessage(uid="valid-agent-uid", iid="also-wrong-iid", data=b"test"),
        # Invalid cancel - wrong uid AND wrong iid
        MultiplexCancelMessage(uid="unknown-uid", iid="bad-iid"),
    ]

    mux = create_multiplexer_with_valid_agent(
        mock_server_session_ctx_with_valid_agent, messages, captured_messages
    )
    await mux.run()

    # Filter to just NewIIDResponse and ErrorMessage (ignore invocation events)
    responses = [
        m
        for m in captured_messages
        if isinstance(m, (MultiplexNewIIDResponse, MultiplexErrorMessage))
    ]

    assert len(responses) == 4

    # First: valid invoke gets NewIIDResponse
    assert isinstance(responses[0], MultiplexNewIIDResponse)
    assert responses[0].match_id == "valid-invoke"

    # Second: cancel with wrong iid gets NotRunningError
    assert isinstance(responses[1], MultiplexErrorMessage)
    assert responses[1].iid == "wrong-iid"
    assert "notrunningerror" in responses[1].error_name.lower()

    # Third: data with wrong iid gets NotRunningError
    assert isinstance(responses[2], MultiplexErrorMessage)
    assert responses[2].iid == "also-wrong-iid"
    assert "notrunningerror" in responses[2].error_name.lower()

    # Fourth: cancel with wrong uid AND iid gets NotRunningError
    assert isinstance(responses[3], MultiplexErrorMessage)
    assert responses[3].iid == "bad-iid"
    assert "notrunningerror" in responses[3].error_name.lower()
