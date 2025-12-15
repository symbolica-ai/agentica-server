from unittest.mock import Mock

import pytest
from openai.types.chat import *

from com.conversion.openai_chat_completions.anthropic import *
from com.deltas import *
from com.roles import *


class TestMakeTextBlock:
    def test_make_text_block_basic(self):
        content = "Hello, world!"
        result = make_text_block(content)

        assert result == ChatCompletionContentPartTextParam(type='text', text=content)
        assert result['type'] == 'text'
        assert result['text'] == content

    def test_make_text_block_empty_string(self):
        content = ""
        result = make_text_block(content)

        assert result == ChatCompletionContentPartTextParam(type='text', text=content)
        assert result['text'] == ""

    def test_make_text_block_multiline(self):
        content = "Line 1\nLine 2\nLine 3"
        result = make_text_block(content)

        assert result['text'] == content


class TestMakeSystemBlock:
    def test_make_system_block_valid_content(self):
        delta = Delta(id="test", name=SystemRole(), content="System message")
        result = make_system_block(delta)

        assert result == ChatCompletionContentPartTextParam(type='text', text="System message")

    def test_make_system_block_none_content_raises_error(self):
        delta = Delta(id="test", name=SystemRole(), content=None)

        with pytest.raises(ValueError, match="System message content is required"):
            make_system_block(delta)

    def test_make_system_block_empty_content(self):
        delta = Delta(id="test", name=SystemRole(), content="")
        result = make_system_block(delta)

        assert result['text'] == ""


class TestMakeUserBlock:
    def test_make_user_block_valid_content(self):
        delta = Delta(id="test", name=UserRole(), content="User message")
        result = make_user_block(delta)

        assert result == ChatCompletionContentPartTextParam(type='text', text="User message")

    def test_make_user_block_none_content_raises_error(self):
        delta = Delta(id="test", name=UserRole(), content=None)

        with pytest.raises(ValueError, match="User message content is required"):
            make_user_block(delta)

    def test_make_user_block_with_username_raises_error(self):
        delta = Delta(id="test", name=UserRole(username="alice"), content="User message")

        with pytest.raises(ValueError, match="Anthropic does not support names in user messages"):
            make_user_block(delta)

    def test_make_user_block_without_username(self):
        delta = Delta(id="test", name=UserRole(username=None), content="User message")
        result = make_user_block(delta)

        assert result['text'] == "User message"


class TestMakeAssistantTextBlock:
    def test_make_assistant_text_block_with_content(self):
        delta = Delta(id="test", name=AgentRole(), content="Assistant message")
        result = make_assistant_text_block(delta)

        assert result == ChatCompletionContentPartTextParam(type='text', text="Assistant message")

    def test_make_assistant_text_block_none_content(self):
        delta = Delta(id="test", name=AgentRole(), content=None)
        result = make_assistant_text_block(delta)

        assert result == ChatCompletionContentPartTextParam(type='text', text=None)


class TestMakeToolTextBlock:
    def test_make_tool_text_block_string_result(self):
        result = make_tool_text_block("Tool result")

        assert result == ChatCompletionContentPartTextParam(type='text', text="Tool result")

    def test_make_tool_text_block_non_string_result(self):
        result = make_tool_text_block({"key": "value"})

        assert result == ChatCompletionContentPartTextParam(type='text', text="{'key': 'value'}")

    def test_make_tool_text_block_number_result(self):
        result = make_tool_text_block(42)

        assert result == ChatCompletionContentPartTextParam(type='text', text="42")

    def test_make_tool_text_block_none_result(self):
        result = make_tool_text_block(None)

        assert result == ChatCompletionContentPartTextParam(type='text', text="None")


class TestMakeToolResult:
    def test_make_tool_result_string_result(self):
        result = make_tool_result("tool_id_123", "Success")

        expected = ChatCompletionToolMessageParam(
            role='tool',
            content=[ChatCompletionContentPartTextParam(type='text', text="Success")],
            tool_call_id="tool_id_123",
        )
        assert result == expected

    def test_make_tool_result_non_string_result(self):
        result = make_tool_result("tool_id_456", {"status": "ok"})

        expected = ChatCompletionToolMessageParam(
            role='tool',
            content=[ChatCompletionContentPartTextParam(type='text', text="{'status': 'ok'}")],
            tool_call_id="tool_id_456",
        )
        assert result == expected


# class TestMakeAssistantToolResults:
#     def test_make_assistant_tool_results_with_results(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
#         )
#
#         results = make_assistant_tool_results(end)
#
#         assert len(results) == 1
#         assert results[0]['role'] == 'tool'
#         assert results[0]['tool_call_id'] == "tool_id_1"
#         assert len(results[0]['content']) == 1
#         assert results[0]['content'][0]['text'] == "result1"
#
#     def test_make_assistant_tool_results_multiple_same_id(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint, mock_constraint],
#             ids=["tool_id_1", "tool_id_1"],
#             content=["arg1", "arg2"],
#             results=["result1", "result2"],
#         )
#
#         results = make_assistant_tool_results(end)
#
#         assert len(results) == 1
#         assert results[0]['tool_call_id'] == "tool_id_1"
#         assert len(results[0]['content']) == 2
#         assert results[0]['content'][0]['text'] == "result1"
#         assert results[0]['content'][1]['text'] == "result2"
#
#     def test_make_assistant_tool_results_different_ids(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint, mock_constraint],
#             ids=["tool_id_1", "tool_id_2"],
#             content=["arg1", "arg2"],
#             results=["result1", "result2"],
#         )
#
#         results = make_assistant_tool_results(end)
#
#         assert len(results) == 2
#         assert results[0]['tool_call_id'] == "tool_id_1"
#         assert results[1]['tool_call_id'] == "tool_id_2"
#
#     def test_make_assistant_tool_results_no_results(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=None
#         )
#
#         results = make_assistant_tool_results(end)
#
#         assert results == []


class TestMakeAssistantToolBlocks:
    def test_make_assistant_tool_blocks_regular_delta(self):
        delta = Delta(id="test", name=AgentRole(), content="Assistant message")

        text_block, tool_calls, tool_results = make_assistant_tool_blocks(delta)

        assert text_block == ChatCompletionContentPartTextParam(
            type='text', text="Assistant message"
        )
        assert tool_calls == []
        assert tool_results == []

    # @patch('com.conversion.openai_chat_completions.anthropic.make_assistant_tool_call_blocks')
    # def test_make_assistant_tool_blocks_generated_delta_with_callable_end(
    #     self, mock_make_tool_calls
    # ):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_call = Mock(spec=ChatCompletionMessageToolCallParam)
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     text_block, tool_calls, tool_results = make_assistant_tool_blocks(delta)
    #
    #     assert text_block == ChatCompletionContentPartTextParam(
    #         type='text', text="Assistant message"
    #     )
    #     assert tool_calls == [mock_tool_call]
    #     assert len(tool_results) == 1
    #     assert tool_results[0]['tool_call_id'] == "tool_id_1"


class TestMakeSystem:
    def test_make_system_valid_content(self):
        delta = Delta(id="test", name=SystemRole(), content="System message")
        result = make_system(delta)

        expected = ChatCompletionSystemMessageParam(
            content=[ChatCompletionContentPartTextParam(type='text', text="System message")],
            role='system',
        )
        assert result == expected

    def test_make_system_none_content_raises_error(self):
        delta = Delta(id="test", name=SystemRole(), content=None)

        with pytest.raises(ValueError, match="System message content is required"):
            make_system(delta)


class TestMakeUser:
    def test_make_user_valid_content(self):
        delta = Delta(id="test", name=UserRole(), content="User message")
        result = make_user(delta)

        expected = ChatCompletionUserMessageParam(
            content=[ChatCompletionContentPartTextParam(type='text', text="User message")],
            role='user',
        )
        assert result == expected


class TestMakeAssistantTool:
    def test_make_assistant_tool_regular_delta(self):
        delta = Delta(id="test", name=AgentRole(), content="Assistant message")
        result = make_assistant_tool(delta)

        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert isinstance(result[0]['content'], list)
        assert len(result[0]['content']) == 1
        assert result[0]['content'][0] == ChatCompletionContentPartTextParam(
            type='text', text="Assistant message"
        )
        assert 'tool_calls' not in result[0]

    # @patch('com.conversion.openai_chat_completions.anthropic.make_assistant_tool_call_blocks')
    # def test_make_assistant_tool_with_callable_end(self, mock_make_tool_calls):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
    #     )
    #
    #     delta = GeneratedDelta(id="test", name=AgentRole(), content="Assistant message", end=end)
    #
    #     mock_tool_call = Mock(spec=ChatCompletionMessageToolCallParam)
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     result = make_assistant_tool(delta)
    #
    #     assert len(result) == 2  # Assistant message + tool result
    #     assert result[0]['role'] == 'assistant'
    #     assert result[0]['tool_calls'] == [mock_tool_call]
    #     assert result[1]['role'] == 'tool'
    #     assert result[1]['tool_call_id'] == "tool_id_1"


class TestConvertFromRole:
    def test_convert_from_role_generated_delta(self):
        delta = GeneratedDelta(id="test", name=AgentRole())
        result = convert_from_role(delta)

        assert result == 'assistant'

    def test_convert_from_role_user_role(self):
        delta = Delta(id="test", name=UserRole())
        result = convert_from_role(delta)

        assert result == 'user'

    def test_convert_from_role_agent_role(self):
        delta = Delta(id="test", name=AgentRole())
        result = convert_from_role(delta)

        assert result == 'assistant'

    def test_convert_from_role_system_role(self):
        delta = Delta(id="test", name=SystemRole())
        result = convert_from_role(delta)

        assert result == 'system'

    def test_convert_from_role_invalid_role(self):
        delta = Delta(id="test", name=Mock())  # Invalid role

        with pytest.raises(ValueError, match="Invalid role"):
            convert_from_role(delta)


class TestConvertFromDeltas:
    def test_convert_from_deltas_system_only(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="System message"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert len(result[0]['content']) == 1
        assert result[0]['content'][0]['text'] == "System message"

    def test_convert_from_deltas_user_message(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="Hello"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1  # Just user message
        assert result[0]['role'] == 'user'
        assert result[0]['content'][0]['text'] == "Hello"

    def test_convert_from_deltas_assistant_message(self):
        deltas = [
            GeneratedDelta(id="1", name=AgentRole(), content="Hi there"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1  # Just assistant message
        assert isinstance(result[0], dict)
        assert result[0]['role'] == 'assistant'

    def test_convert_from_deltas_multiple_system_messages_concatenated(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="System part 1"),
            Delta(id="2", name=SystemRole(), content="System part 2"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1
        assert result[0]['role'] == 'system'
        # Now with the fix, system messages should properly concatenate
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['text'] == "System part 1"
        assert result[0]['content'][1]['text'] == "System part 2"

    def test_convert_from_deltas_multiple_user_messages_concatenated(self):
        deltas = [
            Delta(id="1", name=UserRole(), content="User part 1"),
            Delta(id="2", name=UserRole(), content="User part 2"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 1  # Just user message
        assert result[0]['role'] == 'user'
        assert len(result[0]['content']) == 2
        assert result[0]['content'][0]['text'] == "User part 1"
        assert result[0]['content'][1]['text'] == "User part 2"

    def test_convert_from_deltas_role_switching(self):
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

    # @patch('com.conversion.openai_chat_completions.anthropic.make_assistant_tool_call_blocks')
    # def test_convert_from_deltas_assistant_with_tool_calls(self, mock_make_tool_calls):
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #
    #     end = GenEndCallableTypes(
    #         constraints=[mock_constraint], ids=["tool_id_1"], content=["arg1"], results=["result1"]
    #     )
    #
    #     deltas = [
    #         GeneratedDelta(id="1", name=AgentRole(), content="Calling tool", end=end),
    #     ]
    #
    #     mock_tool_call = Mock(spec=ChatCompletionMessageToolCallParam)
    #     mock_make_tool_calls.return_value = [mock_tool_call]
    #
    #     result = convert_from_deltas(deltas)
    #
    #     # Should have assistant message + tool result
    #     assert len(result) == 2  # Assistant message + tool result
    #     assert result[0]['role'] == 'assistant'
    #     assert result[1]['role'] == 'tool'

    def test_convert_from_deltas_empty_list(self):
        result = convert_from_deltas([])

        assert len(result) == 0

    def test_convert_from_deltas_complex_conversation(self):
        deltas = [
            Delta(id="1", name=SystemRole(), content="You are helpful"),
            Delta(id="2", name=SystemRole(), content=" and friendly"),
            Delta(id="3", name=UserRole(), content="Hello"),
            GeneratedDelta(id="4", name=AgentRole(), content="Hi there!"),
            Delta(id="5", name=UserRole(), content="How are you?"),
            GeneratedDelta(id="6", name=AgentRole(), content="I'm doing well"),
        ]

        result = convert_from_deltas(deltas)

        assert len(result) == 5
        # System message with 2 content blocks (now properly concatenated)
        assert result[0]['role'] == 'system'
        assert len(result[0]['content']) == 2
        # First user message
        assert result[1]['role'] == 'user'
        # First assistant message
        assert result[2]['role'] == 'assistant'
        # Second user message
        assert result[3]['role'] == 'user'
        # Second assistant message
        assert result[4]['role'] == 'assistant'
