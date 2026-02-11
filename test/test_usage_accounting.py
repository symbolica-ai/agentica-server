"""Tests for desired usage accounting behaviour at the ServerSessionManager level.

Uses the real ServerSessionManager, its create_agent() path, a real Agent,
and the exact notifier chain the Multiplexer builds — only the LLM call
(InferenceSystem.invoke) is mocked.

"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agentica_internal.session_manager_messages import AllServerMessage
from agentica_internal.session_manager_messages.session_manager_messages import (
    CreateAgentRequest,
    SMInferenceUsageMessage,
)

from com.actions_model import ModelInference
from com.gen_model import Generation
from messages import HybridNotifier, Notifier
from messages.poster import Poster
from server_session_manager.server_session_manager import ServerSessionManager

CID = "test-session"
IID = "invocation-0"

AGENT_REQUEST = CreateAgentRequest(
    doc=None,
    system=None,
    model="openai:gpt-4o",
    streaming=False,
    warp_globals_payload=b"",
    protocol="python/v0",
)


def _make_mock_response_usage() -> MagicMock:
    """Mock that satisfies GenAIUsage.from_response_usage() and
    json.dumps(response.usage.model_dump()) in try_inference."""
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.total_tokens = 150
    usage.input_tokens_details = MagicMock(cached_tokens=0)
    usage.output_tokens_details = MagicMock(reasoning_tokens=0)
    usage.model_dump.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return usage


def _create_ssm(mock_system: MagicMock) -> ServerSessionManager:
    """Create a real SSM with a mock provider that returns *mock_system*."""
    mock_provider = MagicMock()
    mock_provider.create_inference_system.return_value = mock_system
    return ServerSessionManager(
        log_poster=Poster(url="http://localhost:0/noop"),
        providers=[mock_provider],
        silent_for_testing=True,
    )


def _build_notifier_chain(ssm: ServerSessionManager, uid: str) -> HybridNotifier:
    """Replicate the per-agent notifier chain that Multiplexer._create_agent_context
    builds, pointing at the SSM's real Holder."""

    async def noop_send(_: object) -> None:
        pass

    legacy_notifier = Notifier(
        uid=uid,
        send_mx_message=noop_send,
        log_poster=ssm.log_poster,
        logs=ssm._logs,  # the real SSM Holder
    )
    return HybridNotifier(
        uid=uid,
        send_mx_message=noop_send,
        legacy_notifier=legacy_notifier,
        otel_notifier=None,
    )


def _wire_invocation(ssm: ServerSessionManager, uid: str, hybrid: HybridNotifier) -> None:
    """Replicate Agent._setup_callbacks wiring for a single invocation."""
    agent = ssm._agents[uid]
    inv = hybrid.bind_invocation(IID)

    async def monad_log(body: str) -> None:
        await inv.log_monad(body)

    agent.inference_context.monad_log = monad_log
    agent.inference_context.invocation = inv
    agent.inference_context.inference_config.iid = IID
    inv.with_agent_metadata(model=agent.model.identifier, provider=agent.model.provider)


def _is_usage(msg: AllServerMessage) -> bool:
    return isinstance(msg, SMInferenceUsageMessage)


ROUNDS = 300


@pytest.mark.asyncio
async def test_listen_vs_get_logs():
    """Run inference rounds through a real Agent created by a real
    ServerSessionManager. ssm.get_logs() and the ssm.listen() should
    see all usage messages."""

    mock_usage = _make_mock_response_usage()
    generation = Generation(output_text="answer", code=None, usage=mock_usage)
    mock_system = MagicMock()
    mock_system.uses_tool_calls = False
    mock_system.has_messages = True
    mock_system.invoke = AsyncMock(return_value=generation)

    ssm = _create_ssm(mock_system)
    ssm.register_session(CID)
    uid = await ssm.create_agent(AGENT_REQUEST, CID)
    agent = ssm._agents[uid]

    hybrid = _build_notifier_chain(ssm, uid)
    _wire_invocation(ssm, uid, hybrid)

    listener_captured: list[AllServerMessage] = []
    ssm._logs.add_listener(uid, listener_captured.append)

    # ModelInference.try_inference with the Agent's context
    action = ModelInference()
    for _ in range(ROUNDS):
        await action.try_inference(agent.inference_context)

    # Listener (ssm.listen() / echo stream / log-file path)
    listener_usage = [m for m in listener_captured if _is_usage(m)]
    assert len(listener_usage) == ROUNDS, (
        f"/echo path should capture all {ROUNDS} usage messages, but got {len(listener_usage)}"
    )

    # SSM (_fetch_usage / get_logs)
    ssm_usage = list(ssm.get_logs(uid, IID, _is_usage))
    assert len(ssm_usage) == ROUNDS, (
        f"ssm.get_logs() should return all {ROUNDS} usage messages, but only got {len(ssm_usage)}"
    )

    ssm.close()


@pytest.mark.asyncio
async def test_usage_available_immediately_after_inference():
    """Run one inference round through the real Agent created by a real
    ServerSessionManager.  The usage message must be available in
    ssm.get_logs() immediately."""

    mock_usage = _make_mock_response_usage()
    generation = Generation(output_text="answer", code=None, usage=mock_usage)
    mock_system = MagicMock()
    mock_system.uses_tool_calls = False
    mock_system.has_messages = True
    mock_system.invoke = AsyncMock(return_value=generation)

    ssm = _create_ssm(mock_system)
    ssm.register_session(CID)
    uid = await ssm.create_agent(AGENT_REQUEST, CID)

    hybrid = _build_notifier_chain(ssm, uid)
    _wire_invocation(ssm, uid, hybrid)

    # One round of try_inference
    await ModelInference().try_inference(ssm._agents[uid].inference_context)

    # SSM (_fetch_usage / get_logs)
    usage_msgs = list(ssm.get_logs(uid, IID, _is_usage))
    assert len(usage_msgs) == 1, (
        f"ssm.get_logs() should return 1 usage message immediately after "
        f"try_inference(), got {len(usage_msgs)}"
    )

    ssm.close()
