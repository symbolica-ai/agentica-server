from unittest.mock import Mock

import pytest
from openai.types.chat import (
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from com.conversion.openai_chat_completions.anthropic import *
from com.conversion.openai_chat_completions.openai import (
    convert_from_delta,
    convert_from_deltas,
    convert_from_role,
    make_assistant,
    make_developer,
    make_system,
    make_user,
)
from com.deltas import *
from com.roles import *


class TestMakeSystem:
    def test_make_system_valid_content(self):
        delta = Delta(id="test", name=SystemRole(), content="System message")
        result = make_system(delta)

        expected = ChatCompletionSystemMessageParam(
            content="System message",
            role='system',
        )
        assert result == expected

    def test_make_system_none_content_raises_error(self):
        delta = Delta(id="test", name=SystemRole(), content=None)

        with pytest.raises(ValueError, match="System message content is required"):
            make_system(delta)

    def test_make_system_empty_content(self):
        delta = Delta(id="test", name=SystemRole(), content="")
        result = make_system(delta)

        assert result['content'] == ""
        assert result['role'] == 'system'


class TestMakeUser:
    def test_make_user_valid_content(self):
        delta = Delta(id="test", name=UserRole(), content="User message")
        result = make_user(delta)

        expected = ChatCompletionUserMessageParam(
            content="User message",
            role='user',
        )
        assert result == expected

    def test_make_user_none_content_raises_error(self):
        delta = Delta(id="test", name=UserRole(), content=None)

        with pytest.raises(ValueError, match="User message content is required"):
            make_user(delta)

    def test_make_user_with_username(self):
        delta = Delta(id="test", name=UserRole(username="alice"), content="User message")
        result = make_user(delta)

        assert result['content'] == "User message"
        assert result['role'] == 'user'
        assert result['name'] == "alice"

    def test_make_user_without_username(self):
        delta = Delta(id="test", name=UserRole(username=None), content="User message")
        result = make_user(delta)

        assert result['content'] == "User message"
        assert result['role'] == 'user'
        assert 'name' not in result

    def test_make_user_empty_content(self):
        delta = Delta(id="test", name=UserRole(), content="")
        result = make_user(delta)

        assert result['content'] == ""


class TestMakeAssistant:
    def test_make_assistant_regular_delta(self):
        delta = Delta(id="test", name=AgentRole(), content="Assistant message")
        result = make_assistant(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == "Assistant message"
        assert result[0]['refusal'] is None

    def test_make_assistant_generated_delta_with_refusal(self):
        delta = GeneratedDelta(
            id="test",
            name=AgentRole(),
            content="Assistant message",
            refusal="I cannot help with that",
        )
        result = make_assistant(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == "Assistant message"
        assert result[0]['refusal'] == "I cannot help with that"

    def test_make_assistant_generated_delta_with_audio(self):
        delta = GeneratedDelta(
            id="test", name=AgentRole(), content="Assistant message", audio={"id": "audio_123"}
        )
        result = make_assistant(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['audio'] == {"id": "audio_123"}

    def test_make_assistant_generated_delta_with_invalid_audio(self):
        delta = GeneratedDelta(
            id="test", name=AgentRole(), content="Assistant message", audio={"invalid": "data"}
        )

        with pytest.raises(ValueError, match="Audio message must have an string id"):
            make_assistant(delta)

    def test_make_assistant_generated_delta_with_audio_non_string_id(self):
        delta = GeneratedDelta(
            id="test", name=AgentRole(), content="Assistant message", audio={"id": 123}
        )

        with pytest.raises(ValueError, match="Audio message must have an string id"):
            make_assistant(delta)

    # @patch('com.conversion.openai_chat_completions.openai.make_assistant_tool_call_blocks')
    # def test_make_assistant_with_callable_end(self, mock_make_tool_calls):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_call = Mock()
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     result = make_assistant(delta)
    #
    #     assert len(result) == 2  # Assistant message + tool result
    #     assert result[0]['role'] == 'assistant'
    #     assert result[0]['tool_calls'] == [mock_tool_call]
    #     assert result[1]['role'] == 'tool'
    #     assert result[1]['content'] == "result1"
    #     assert result[1]['tool_call_id'] == "tool_id_1"

    # @patch('com.conversion.openai_chat_completions.openai.make_assistant_tool_call_blocks')
    # def test_make_assistant_with_callable_end_no_results(self, mock_make_tool_calls):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=None
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_call = Mock()
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     result = make_assistant(delta)
    #
    #     assert len(result) == 1  # Only assistant message
    #     assert result[0]['role'] == 'assistant'
    #     assert result[0]['tool_calls'] == [mock_tool_call]

    # @patch('com.conversion.openai_chat_completions.openai.make_assistant_tool_call_blocks')
    # def test_make_assistant_with_callable_end_multiple_results(self, mock_make_tool_calls):
    #     mock_constraint1 = Mock(spec=CallableConstraint)
    #     mock_constraint1.name = "func1"
    #     mock_constraint2 = Mock(spec=CallableConstraint)
    #     mock_constraint2.name = "func2"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint1, mock_constraint2],
    #         ids=["tool_id_1", "tool_id_2"],
    #         content=["arg1", "arg2"],
    #         results=["result1", "result2"],
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_calls = [Mock(), Mock()]
    #     mock_make_tool_calls.return_value = mock_tool_calls
    #
    #     result = make_assistant(delta)
    #
    #     assert len(result) == 3  # Assistant message + 2 tool results
    #     assert result[0]['role'] == 'assistant'
    #     assert result[0]['tool_calls'] == mock_tool_calls
    #     assert result[1]['role'] == 'tool'
    #     assert result[1]['content'] == "result1"
    #     assert result[1]['tool_call_id'] == "tool_id_1"
    #     assert result[2]['role'] == 'tool'
    #     assert result[2]['content'] == "result2"
    #     assert result[2]['tool_call_id'] == "tool_id_2"
    #
    # @patch('com.conversion.openai_chat_completions.openai.make_assistant_tool_call_blocks')
    # def test_make_assistant_with_callable_end_non_string_result(self, mock_make_tool_calls):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint],
    #         ids=["tool_id_1"],
    #         content=["arg1"],
    #         results=[{"key": "value"}],
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_call = Mock()
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     result = make_assistant(delta)
    #
    #     assert len(result) == 2
    #     assert result[1]['content'] == "{'key': 'value'}"


class TestMakeDeveloper:
    def test_make_developer_valid_content(self):
        delta = Delta(id="test", name=SystemRole(), content="Developer message")
        result = make_developer(delta)

        expected = ChatCompletionDeveloperMessageParam(
            content="Developer message",
            role='developer',
        )
        assert result == expected

    def test_make_developer_none_content_raises_error(self):
        delta = Delta(id="test", name=SystemRole(), content=None)

        with pytest.raises(ValueError, match="Developer message content is required"):
            make_developer(delta)

    def test_make_developer_empty_content(self):
        delta = Delta(id="test", name=SystemRole(), content="")
        result = make_developer(delta)

        assert result['content'] == ""
        assert result['role'] == 'developer'


class TestConvertFromDelta:
    def test_convert_from_delta_generated_delta(self):
        delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message")
        result = convert_from_delta(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == "Assistant message"

    def test_convert_from_delta_system_role(self):
        delta = Delta(id="test", name=SystemRole(), content="System message")
        result = convert_from_delta(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == "System message"

    def test_convert_from_delta_agent_role(self):
        delta = Delta(id="test", name=AgentRole(), content="Assistant message")
        result = convert_from_delta(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == "Assistant message"

    def test_convert_from_delta_user_role(self):
        delta = Delta(id="test", name=UserRole(), content="User message")
        result = convert_from_delta(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == "User message"

    def test_convert_from_delta_invalid_role(self):
        delta = Delta(id="test", name=Mock(), content="Invalid message")

        with pytest.raises(ValueError, match="Invalid delta"):
            convert_from_delta(delta)


class TestConvertFromRole:
    def test_convert_from_role_user_role(self):
        role = UserRole()
        result = convert_from_role(role)

        assert result == 'user'

    def test_convert_from_role_agent_role(self):
        role = AgentRole()
        result = convert_from_role(role)

        assert result == 'assistant'

    def test_convert_from_role_system_role(self):
        role = SystemRole()
        result = convert_from_role(role)

        assert result == 'system'

    def test_convert_from_role_invalid_role(self):
        role = Mock()

        with pytest.raises(ValueError, match="Invalid role"):
            convert_from_role(role)


class TestConvertFromDeltas:
    def test_convert_from_deltas_empty_list(self):
        result = convert_from_deltas([])

        assert result == []

    def test_convert_from_deltas_single_delta(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="System message"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == "System message"

    def test_convert_from_deltas_different_roles(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="System message"),
            Delta(id="2", name=UserRole(), content="User message"),
            GeneratedDelta(id="3", name=AgentRole(), content="Assistant message"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 3
        assert result[0]['role'] == 'system'
        assert result[1]['role'] == 'user'
        assert result[2]['role'] == 'assistant'

    def test_convert_from_deltas_same_role_concatenation(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="Hello"),
            Delta(id="2", name=UserRole(), content=" world"),
            Delta(id="3", name=UserRole(), content="!"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == "Hello world!"

    def test_convert_from_deltas_same_role_with_none_content(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="Hello"),
            Delta(id="2", name=UserRole(), content=None),
            Delta(id="3", name=UserRole(), content=" world"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == "Hello world"

    def test_convert_from_deltas_generated_delta_reasoning_content(self):
        deltas = [
            GeneratedDelta(
                id="1", name=AgentRole(), content="Answer", reasoning_content="Let me think"
            ),
            GeneratedDelta(
                id="2", name=AgentRole(), content=" here", reasoning_content=" about this"
            ),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        # Content should include both content and reasoning_content concatenated
        assert result[0]['content'] == "AnswerLet me think here about this"

    def test_convert_from_deltas_generated_delta_reasoning_content_with_none_content(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="User message"),
            GeneratedDelta(
                id="2",
                name=AgentRole(),
                content=None,  # None content
                reasoning_content="This should work now",
            ),
        ]

        # This should work now because content=None gets converted to '' before concatenation
        result = convert_from_deltas(deltas)
        assert len(result) == 2
        assert result[0]['role'] == 'user'
        assert result[1]['role'] == 'assistant'
        assert result[1]['content'] == "This should work now"  # reasoning_content only

    def test_convert_from_deltas_empty_content_handling(self):
        deltas = [
            Delta(id="1", name=UserRole(), content=""),
            Delta(id="2", name=UserRole(), content="Hello"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['content'] == "Hello"

    def test_convert_from_deltas_none_content_handling(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="Hello"),  # First delta must have content
            Delta(id="2", name=UserRole(), content=None),  # Second can be None
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['content'] == "Hello"  # None content is ignored

    def test_convert_from_deltas_complex_conversation(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="You are helpful"),
            Delta(id="2", name=UserRole(), content="Hello"),
            GeneratedDelta(id="3", name=AgentRole(), content="Hi"),
            GeneratedDelta(id="4", name=AgentRole(), content=" there!"),
            Delta(id="5", name=UserRole(), content="How are you?"),
            GeneratedDelta(id="6", name=AgentRole(), content="I'm good"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 5  # System, User, Assistant (concatenated), User, Assistant
        # System message
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == "You are helpful"
        # First user message
        assert result[1]['role'] == 'user'
        assert result[1]['content'] == "Hello"
        # First assistant message (concatenated)
        assert result[2]['role'] == 'assistant'
        assert result[2]['content'] == "Hi there!"
        # Second user message
        assert result[3]['role'] == 'user'
        assert result[3]['content'] == "How are you?"
        # Second assistant message
        assert result[4]['role'] == 'assistant'
        assert result[4]['content'] == "I'm good"

    # def test_convert_from_deltas_with_tool_calls(self):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
    #     )
    #
    #     deltas = [
    #         GeneratedDelta(id="1", name=AgentRole(), content="Using tool", end=end),
    #     ]
    #
    #     with patch(
    #         'com.conversion.openai_chat_completions.openai.make_assistant_tool_call_blocks'
    #     ) as mock_make_tool_calls:
    #         mock_tool_call = Mock()
    #         mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #         result = convert_from_deltas(deltas)
    #
    #         assert len(result) == 2  # Assistant message + tool result
    #         assert result[0]['role'] == 'assistant'
    #         assert result[0]['content'] == "Using tool"
    #         assert result[0]['tool_calls'] == [mock_tool_call]
    #         assert result[1]['role'] == 'tool'
    #         assert result[1]['content'] == "result1"

    def test_convert_from_deltas_reasoning_content_with_none_previous(self):
        deltas = [
            GeneratedDelta(id="1", name=AgentRole(), content="Answer", reasoning_content=None),
            GeneratedDelta(id="2", name=AgentRole(), content=" here", reasoning_content="Thinking"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert (
            result[0]['content'] == "Answer hereThinking"
        )  # reasoning_content is added to content

    def test_convert_from_deltas_content_with_none_previous(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="Hello"),  # First delta must have content
            Delta(id="2", name=UserRole(), content=None),  # Second can be None
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['content'] == "Hello"  # None content is ignored
