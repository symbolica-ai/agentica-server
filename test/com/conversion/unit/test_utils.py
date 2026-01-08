from unittest.mock import Mock, patch

import pytest

from com.constraints import *
from com.context import *
from com.conversion.openai_chat_completions.utils import (
    convert_from_constraints,
    convert_to_delta,
    convert_to_role,
    make_end,
)
from com.deltas import *
from com.roles import *

# class TestMakeAssistantToolCallBlocks:
#     def test_make_assistant_tool_call_blocks_single_call(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint], ids=["tool_id_1"], content=[{"arg1": "value1"}]
#         )
#
#         result = make_assistant_tool_call_blocks(end)
#
#         assert len(result) == 1
#         assert result[0]['id'] == "tool_id_1"
#         assert result[0]['function']['name'] == "test_function"
#         assert result[0]['function']['arguments'] == '{"arg1": "value1"}'
#         assert result[0]['type'] == 'function'
#
#     def test_make_assistant_tool_call_blocks_multiple_calls(self):
#         mock_constraint1 = Mock(spec=CallableConstraint)
#         mock_constraint1.name = "function1"
#         mock_constraint2 = Mock(spec=CallableConstraint)
#         mock_constraint2.name = "function2"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint1, mock_constraint2],
#             ids=["tool_id_1", "tool_id_2"],
#             content=[{"arg1": "value1"}, {"arg2": "value2"}],
#         )
#
#         result = make_assistant_tool_call_blocks(end)
#
#         assert len(result) == 2
#         assert result[0]['function']['name'] == "function1"
#         assert result[1]['function']['name'] == "function2"
#         assert result[0]['function']['arguments'] == '{"arg1": "value1"}'
#         assert result[1]['function']['arguments'] == '{"arg2": "value2"}'
#
#     def test_make_assistant_tool_call_blocks_string_content(self):
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#
#         end = GenEndCallableTypes(
#             constraints=[mock_constraint], ids=["tool_id_1"], content=['{"already": "json"}']
#         )
#
#         result = make_assistant_tool_call_blocks(end)
#
#         assert result[0]['function']['arguments'] == '{"already": "json"}'
#
#     def test_make_assistant_tool_call_blocks_empty_end(self):
#         end = GenEndCallableTypes(constraints=[], ids=[], content=[])
#
#         result = make_assistant_tool_call_blocks(end)
#
#         assert result == []


# class TestMakeToolDef:
#     @pytest.mark.asyncio
#     async def test_make_tool_def_basic(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'callable'
#         mock_executable.name = "test_function"
#         mock_executable.description = "A test function"
#
#         mock_sandbox = Mock(spec=Sandbox)
#
#         async def test_function_schema(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object", "properties": {"arg1": {"type": "string"}}}
#
#         mock_executable.schema = test_function_schema
#
#         result = await make_tool_def(mock_executable, mock_sandbox, guided=False)
#
#         expected = ChatCompletionToolParam(
#             type='function',
#             function=FunctionDefinition(
#                 name="test_function",
#                 description="A test function",
#                 parameters={
#                     "type": "object",
#                     "properties": {"arg1": {"type": "string"}},
#                     "additionalProperties": False,
#                 },
#                 strict=False,
#             ),
#         )
#         assert result == expected
#
#     @pytest.mark.asyncio
#     async def test_make_tool_def_guided(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'callable'
#         mock_executable.name = "test_function"
#         mock_executable.description = "A test function"
#         mock_sandbox = Mock(spec=Sandbox)
#
#         async def test_function_schema(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object"}
#
#         mock_executable.schema = test_function_schema
#
#         result = await make_tool_def(mock_executable, mock_sandbox, guided=True)
#
#         expected = ChatCompletionToolParam(
#             type='function',
#             function=FunctionDefinition(
#                 name="test_function",
#                 description="A test function",
#                 parameters={
#                     "type": "object",
#                     "additionalProperties": False,
#                 },
#                 strict=True,
#             ),
#         )
#         assert result == expected
#
#     @pytest.mark.asyncio
#     async def test_make_tool_def_preserves_existing_additional_properties(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'callable'
#         mock_executable.name = "test_function"
#         mock_executable.description = "A test function"
#
#         mock_sandbox = Mock(spec=Sandbox)
#
#         async def test_function_schema(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object", "additionalProperties": True}
#
#         mock_executable.schema = test_function_schema
#
#         result = await make_tool_def(mock_executable, mock_sandbox, guided=False)
#
#         # Should override additionalProperties to False
#         assert result['function']['parameters']['additionalProperties'] is False
#
#
# class TestMakeNamedToolChoice:
#     @pytest.mark.asyncio
#     async def test_make_named_tool_choice_basic(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'callable'
#         mock_executable.name = "test_function"
#
#         result = make_named_tool_choice(mock_executable)
#
#         expected = ChatCompletionNamedToolChoiceParam(
#             type='function',
#             function=Function(name="test_function"),
#         )
#         assert result == expected
#
#
# class TestMakeNamedJsonSchema:
#     @pytest.mark.asyncio
#     async def test_make_named_json_schema_single_executable(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'object'
#         mock_executable.name = "TestClass"
#         mock_executable.description = "A test class"
#
#         mock_sandbox = Mock(spec=Sandbox)
#
#         async def test_object_schema(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object", "properties": {"field1": {"type": "string"}}}
#
#         mock_executable.schema = test_object_schema
#
#         result = await make_named_json_schema([mock_executable], mock_sandbox, guided=False)
#
#         assert result['type'] == 'json_schema'
#         assert result['json_schema']['name'] == "TestClass"
#         assert result['json_schema']['description'] == "A test class"
#         assert result['json_schema']['schema']['additionalProperties'] is False
#         assert result['json_schema']['strict'] is False
#
#     @pytest.mark.asyncio
#     async def test_make_named_json_schema_multiple_executables(self):
#         mock_sandbox = Mock(spec=Sandbox)
#
#         mock_executable1 = Mock(spec=Executable)
#         mock_executable1.type = 'object'
#         mock_executable1.name = "Class1"
#         mock_executable1.description = "First class"
#
#         async def test_object_schema1(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object", "properties": {"field1": {"type": "string"}}}
#
#         mock_executable1.schema = test_object_schema1
#
#         mock_executable2 = Mock(spec=Executable)
#         mock_executable2.type = 'object'
#         mock_executable2.name = "Class2"
#         mock_executable2.description = "Second class"
#
#         async def test_object_schema2(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object", "properties": {"field2": {"type": "number"}}}
#
#         mock_executable2.schema = test_object_schema2
#
#         result = await make_named_json_schema(
#             [mock_executable1, mock_executable2], mock_sandbox, guided=False
#         )
#
#         assert result['json_schema']['name'] == "Class1 Class2"
#         assert (
#             result['json_schema']['description']
#             == "Any of the following:\n- First class\n- Second class"
#         )
#         assert 'anyOf' in result['json_schema']['schema']
#         assert len(result['json_schema']['schema']['anyOf']) == 2
#
#     @pytest.mark.asyncio
#     async def test_make_named_json_schema_guided(self):
#         mock_executable = Mock(spec=Executable)
#         mock_executable.type = 'object'
#         mock_executable.name = "TestClass"
#         mock_executable.description = "A test class"
#         mock_sandbox = Mock(spec=Sandbox)
#
#         async def test_object_schema(sandbox: Sandbox) -> dict[str, Any]:
#             return {"type": "object"}
#
#         mock_executable.schema = test_object_schema
#
#         result = await make_named_json_schema([mock_executable], mock_sandbox, guided=True)
#
#         assert result['json_schema']['strict'] is True
#
#     @pytest.mark.asyncio
#     async def test_make_named_json_schema_empty_list_raises_error(self):
#         mock_sandbox = Mock(spec=Sandbox)
#         with pytest.raises(ValueError, match="No executables provided"):
#             await make_named_json_schema([], mock_sandbox, guided=False)
#


class TestConvertFromConstraints:
    @pytest.mark.asyncio
    async def test_convert_from_constraints_empty_constraints(self):
        mock_ctx = Mock(spec=Context)
        mock_gen = Mock()
        mock_gen.type = 'json'
        mock_gen.guided = False
        mock_ctx.gen = mock_gen

        result = await convert_from_constraints(mock_ctx, [], validate=False)

        assert result == {}

    @pytest.mark.asyncio
    async def test_convert_from_constraints_stop_tokens(self):
        mock_ctx = Mock(spec=Context)
        mock_gen = Mock()
        mock_gen.type = 'json'
        mock_gen.guided = False
        mock_ctx.gen = mock_gen

        constraints = [
            StopTokenConstraint(token="STOP"),
            StopTokenConstraint(token="END"),
        ]

        result = await convert_from_constraints(mock_ctx, constraints, validate=False)

        assert result['stop'] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_convert_from_constraints_max_tokens(self):
        mock_ctx = Mock(spec=Context)
        mock_gen = Mock()
        mock_gen.type = 'json'
        mock_gen.guided = False
        mock_ctx.gen = mock_gen

        constraints = [
            MaxTokensConstraint(max_tokens=100),
            MaxTokensConstraint(max_tokens=200),  # Should use first one
        ]

        result = await convert_from_constraints(mock_ctx, constraints, validate=False)

        assert result['max_completion_tokens'] == 100

    # @patch('com.conversion.openai_chat_completions.utils.make_named_json_schema')
    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_object_constraint(self, mock_make_schema):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'json'
    #     mock_gen.guided = False
    #     mock_ctx.gen = mock_gen
    #     mock_exec = Mock()
    #     mock_exec.sandbox = Mock(spec=Sandbox)
    #     mock_ctx.exec = mock_exec
    #
    #     mock_executable = Mock(spec=Executable)
    #     mock_executable.type = 'object'
    #
    #     mock_constraint = Mock(spec=ObjectConstraint)
    #     mock_constraint.executables = [mock_executable]
    #     mock_constraint.guided = False
    #
    #     mock_schema = Mock(spec=ResponseFormatJSONSchema)
    #     mock_make_schema.return_value = mock_schema
    #
    #     constraints = [mock_constraint]
    #
    #     result = await convert_from_constraints(mock_ctx, constraints, validate=False)
    #
    #     assert result['response_format'] == mock_schema
    #     mock_make_schema.assert_called_once_with([mock_executable], mock_exec.sandbox, False)

    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_object_constraint_no_executables(self):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'json'
    #     mock_gen.guided = False
    #     mock_ctx.gen = mock_gen
    #
    #     mock_constraint = Mock(spec=ObjectConstraint)
    #     mock_constraint.executables = []
    #
    #     constraints = [mock_constraint]
    #
    #     result = await convert_from_constraints(mock_ctx, constraints, validate=False)
    #
    #     assert 'response_format' not in result

    # @patch('com.conversion.openai_chat_completions.utils.make_tool_def')
    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_callable_constraints(self, mock_make_tool_def):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'json'
    #     mock_gen.guided = False
    #     mock_ctx.gen = mock_gen
    #     mock_exec = Mock()
    #     mock_exec.sandbox = Mock(spec=Sandbox)
    #     mock_ctx.exec = mock_exec
    #
    #     mock_executable = Mock(spec=Executable)
    #     mock_executable.type = 'callable'
    #     mock_executable.name = "test_function"
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.executable = mock_executable
    #     mock_constraint.guided = True
    #
    #     mock_tool_def = Mock(spec=ChatCompletionToolParam)
    #     mock_make_tool_def.return_value = mock_tool_def
    #
    #     constraints = [mock_constraint]
    #
    #     result = await convert_from_constraints(mock_ctx, constraints, validate=False)
    #
    #     assert result['tools'] == [mock_tool_def]
    #     mock_make_tool_def.assert_called_once_with(mock_executable, mock_exec.sandbox, True)
    #
    # @patch('com.conversion.openai_chat_completions.utils.make_named_tool_choice')
    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_single_callable_constraint_tool_choice(
    #     self, mock_make_tool_choice
    # ):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'json'
    #     mock_gen.guided = False
    #     mock_ctx.gen = mock_gen
    #     mock_exec = Mock()
    #     mock_exec.sandbox = Mock(spec=Sandbox)
    #     mock_ctx.exec = mock_exec
    #
    #     mock_executable = Mock(spec=Executable)
    #     mock_executable.type = 'callable'
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.executable = mock_executable
    #     mock_constraint.guided = False
    #
    #     mock_tool_choice = Mock(spec=ChatCompletionNamedToolChoiceParam)
    #     mock_make_tool_choice.return_value = mock_tool_choice
    #
    #     constraints = [mock_constraint]
    #
    #     with patch('com.conversion.openai_chat_completions.utils.make_tool_def'):
    #         result = await convert_from_constraints(mock_ctx, constraints, validate=False)
    #
    #         assert result['tool_choice'] == mock_tool_choice
    #         mock_make_tool_choice.assert_called_once_with(mock_executable)

    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_multiple_callable_constraints_no_tool_choice(self):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'json'
    #     mock_gen.guided = False
    #     mock_ctx.gen = mock_gen
    #     mock_exec = Mock()
    #     mock_exec.sandbox = Mock(spec=Sandbox)
    #     mock_ctx.exec = mock_exec
    #
    #     mock_executable1 = Mock(spec=Executable)
    #     mock_executable1.type = 'callable'
    #     mock_executable1.name = "function1"
    #     mock_constraint1 = Mock(spec=CallableConstraint)
    #     mock_constraint1.executable = mock_executable1
    #     mock_constraint1.guided = False
    #
    #     mock_executable2 = Mock(spec=Executable)
    #     mock_executable2.type = 'callable'
    #     mock_executable2.name = "function2"
    #     mock_constraint2 = Mock(spec=CallableConstraint)
    #     mock_constraint2.executable = mock_executable2
    #     mock_constraint2.guided = False
    #
    #     constraints = [mock_constraint1, mock_constraint2]
    #
    #     with patch('com.conversion.openai_chat_completions.utils.make_tool_def'):
    #         result = await convert_from_constraints(mock_ctx, constraints, validate=False)
    #
    #         assert 'tool_choice' not in result

    # @pytest.mark.asyncio
    # async def test_convert_from_constraints_guided_non_json_raises_error(self):
    #     mock_ctx = Mock(spec=Context)
    #     mock_gen = Mock()
    #     mock_gen.type = 'custom'
    #     mock_gen.guided = True
    #     mock_ctx.gen = mock_gen

    #     with pytest.raises(ValueError, match="Support for code constraints is not implemented yet"):
    #         await convert_from_constraints(mock_ctx, [], validate=False)

    @pytest.mark.asyncio
    async def test_convert_from_constraints_mixed_constraints(self):
        mock_ctx = Mock(spec=Context)
        mock_gen = Mock()
        mock_gen.type = 'json'
        mock_gen.guided = False
        mock_ctx.gen = mock_gen

        stop_constraint = StopTokenConstraint(token="STOP")
        max_tokens_constraint = MaxTokensConstraint(max_tokens=150)

        constraints = [stop_constraint, max_tokens_constraint]

        result = await convert_from_constraints(mock_ctx, constraints, validate=False)

        assert result['stop'] == ["STOP"]
        assert result['max_completion_tokens'] == 150


# class TestMakeCallableEnd:
#     def test_make_callable_end_function_tool_calls(self):
#         tool_calls = [
#             {
#                 'type': 'function',
#                 'id': 'tool_id_1',
#                 'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'},
#             }
#         ]
#
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#         constraints = [mock_constraint]
#
#         result = make_callable_end(tool_calls, constraints)
#
#         assert len(result.constraints) == 1
#         assert result.constraints[0] == mock_constraint
#         assert result.ids == ['tool_id_1']
#         assert result.content == ['{"arg1": "value1"}']
#
#     def test_make_callable_end_multiple_tool_calls(self):
#         tool_calls = [
#             {
#                 'type': 'function',
#                 'id': 'tool_id_1',
#                 'function': {'name': 'function1', 'arguments': '{"arg1": "value1"}'},
#             },
#             {
#                 'type': 'function',
#                 'id': 'tool_id_2',
#                 'function': {'name': 'function2', 'arguments': '{"arg2": "value2"}'},
#             },
#         ]
#
#         mock_constraint1 = Mock(spec=CallableConstraint)
#         mock_constraint1.name = "function1"
#         mock_constraint2 = Mock(spec=CallableConstraint)
#         mock_constraint2.name = "function2"
#         constraints = [mock_constraint1, mock_constraint2]
#
#         result = make_callable_end(tool_calls, constraints)
#
#         assert len(result.constraints) == 2
#         assert result.ids == ['tool_id_1', 'tool_id_2']
#         assert result.content == ['{"arg1": "value1"}', '{"arg2": "value2"}']
#
#     def test_make_callable_end_non_function_type(self):
#         tool_calls = [
#             {
#                 'type': 'other_type',
#                 'id': 'tool_id_1',
#                 'other_type': {'name': 'test_function', 'input': '{"arg1": "value1"}'},
#             }
#         ]
#
#         mock_constraint = Mock(spec=CallableConstraint)
#         mock_constraint.name = "test_function"
#         constraints = [mock_constraint]
#
#         result = make_callable_end(tool_calls, constraints)
#
#         assert result.content == ['{"arg1": "value1"}']
#
#     def test_make_callable_end_empty_tool_calls(self):
#         # The function expects non-empty lists, so this should raise an error
#         with pytest.raises(ValueError, match="No tool calls provided"):
#             make_callable_end([], [])


class TestMakeEnd:
    # def test_make_end_tool_calls_in_message(self):
    #     choice = {
    #         'message': {
    #             'tool_calls': [
    #                 {
    #                     'type': 'function',
    #                     'id': 'tool_id_1',
    #                     'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'},
    #                 }
    #             ]
    #         },
    #         'finish_reason': 'stop',
    #     }
    #
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #     constraints = [mock_constraint]
    #
    #     result = make_end(choice, constraints)
    #
    #     assert isinstance(result, GenEndCallableTypes)
    #     assert len(result.constraints) == 1
    #     assert result.ids == ['tool_id_1']

    # def test_make_end_finish_reason_tool_calls(self):
    #     choice = {
    #         'message': {
    #             'tool_calls': [
    #                 {
    #                     'type': 'function',
    #                     'id': 'tool_id_1',
    #                     'function': {'name': 'test_function', 'arguments': '{"arg1": "value1"}'},
    #                 }
    #             ]
    #         },
    #         'finish_reason': 'tool_calls',
    #     }
    #
    #     mock_constraint = Mock(spec=CallableConstraint)
    #     mock_constraint.name = "test_function"
    #     constraints = [mock_constraint]
    #
    #     result = make_end(choice, constraints)
    #
    #     assert isinstance(result, GenEndCallableTypes)

    def test_make_end_finish_reason_length(self):
        choice = {'message': {}, 'finish_reason': 'length'}

        mock_constraint = Mock(spec=MaxTokensConstraint)
        constraints = [mock_constraint]

        result = make_end(choice, constraints)

        assert isinstance(result, EndGenMaxTokens)
        assert result.constraint == mock_constraint

    def test_make_end_finish_reason_length_no_constraint(self):
        """Test that finish_reason='length' works even without MaxTokensConstraint.

        This can happen when the model hits its provider-side token limit
        but the caller didn't explicitly set max_tokens.
        """
        choice = {'message': {}, 'finish_reason': 'length'}
        constraints: list = []  # No MaxTokensConstraint

        result = make_end(choice, constraints)

        assert isinstance(result, EndGenMaxTokens)
        assert result.constraint is None

    def test_make_end_finish_reason_stop(self):
        choice = {'message': {}, 'finish_reason': 'stop', 'stop_text': 'END'}

        result = make_end(choice, [])

        assert isinstance(result, EndGenStopToken)
        assert result.constraint.token == 'END'

    def test_make_end_finish_reason_stop_no_stop_text(self):
        choice = {'message': {}, 'finish_reason': 'stop'}

        result = make_end(choice, [])

        assert isinstance(result, EndGenStopToken)
        assert result.constraint.token == ''

    def test_make_end_finish_reason_filter_content(self):
        choice = {'message': {}, 'finish_reason': 'filter_content'}

        result = make_end(choice, [])

        assert isinstance(result, EndGenStopToken)
        assert result.constraint is None
        assert result.filtered is True

    def test_make_end_invalid_finish_reason(self):
        choice = {'message': {}, 'finish_reason': 'invalid_reason'}

        with pytest.raises(ValueError, match="Invalid finish reason: invalid_reason"):
            make_end(choice, [])


class TestConvertToRole:
    def test_convert_to_role_user(self):
        result = convert_to_role('user')

        assert isinstance(result, UserRole)
        assert result.username is None

    def test_convert_to_role_user_with_name(self):
        result = convert_to_role('user', 'alice')

        assert isinstance(result, UserRole)
        assert result.username == 'alice'

    def test_convert_to_role_assistant(self):
        result = convert_to_role('assistant')

        assert isinstance(result, AgentRole)

    def test_convert_to_role_system(self):
        result = convert_to_role('system')

        assert isinstance(result, SystemRole)

    def test_convert_to_role_invalid_role(self):
        with pytest.raises(ValueError, match="Invalid role: invalid"):
            convert_to_role('invalid')

    def test_convert_to_role_non_user_with_name_raises_error(self):
        with pytest.raises(AssertionError, match="Only user messages can have a name"):
            convert_to_role('assistant', 'bot_name')


class TestConvertToDelta:
    def test_convert_to_delta_basic(self):
        choice = {
            'message': {'id': 'msg_123', 'role': 'assistant', 'content': 'Hello world'},
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert isinstance(result, GeneratedDelta)
        assert result.id == 'msg_123'
        assert isinstance(result.name, AgentRole)
        assert result.content == 'Hello world'

    def test_convert_to_delta_with_reasoning_content(self):
        choice = {
            'message': {
                'role': 'assistant',
                'content': 'Hello world',
                'reasoning_content': 'Let me think about this',
            },
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert result.reasoning_content == 'Let me think about this'

    def test_convert_to_delta_with_annotations(self):
        choice = {
            'message': {
                'role': 'assistant',
                'content': 'Hello world',
                'annotations': {'key': 'value'},
            },
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert result.annotations == {'key': 'value'}

    def test_convert_to_delta_with_audio(self):
        choice = {
            'message': {
                'role': 'assistant',
                'content': 'Hello world',
                'audio': {'id': 'audio_123'},
            },
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert result.audio == {'id': 'audio_123'}

    def test_convert_to_delta_with_refusal(self):
        choice = {
            'message': {'role': 'assistant', 'content': None, 'refusal': 'I cannot help with that'},
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert result.refusal == 'I cannot help with that'

    def test_convert_to_delta_with_user_name(self):
        choice = {
            'message': {'role': 'user', 'name': 'alice', 'content': 'Hello'},
            'finish_reason': 'stop',
        }

        result = convert_to_delta(choice, [])

        assert isinstance(result.name, UserRole)
        assert result.name.username == 'alice'

    def test_convert_to_delta_missing_fields(self):
        choice = {'message': {}, 'finish_reason': 'stop'}

        # This should fail because role is required
        with pytest.raises(ValueError, match="Invalid role: None"):
            convert_to_delta(choice, [])

    @patch('com.conversion.openai_chat_completions.utils.make_end')
    def test_convert_to_delta_calls_make_end(self, mock_make_end):
        choice = {'message': {'role': 'assistant', 'content': 'Hello'}, 'finish_reason': 'stop'}

        mock_end = Mock()
        mock_make_end.return_value = mock_end

        constraints = [Mock()]

        result = convert_to_delta(choice, constraints, streaming=False)

        assert result.end == mock_end
        assert result.constraints == constraints
        mock_make_end.assert_called_once_with(choice, constraints, streaming=False)
