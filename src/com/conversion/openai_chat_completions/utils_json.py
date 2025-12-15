import json
from typing import TYPE_CHECKING, Any

from openai.types.chat import (
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    chat_completion_message_tool_call_param,
)
from openai.types.chat.chat_completion_named_tool_choice_param import Function
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from com.constraints import *
from com.context import Context
from com.context_json import *

from .check_schema import OpenAIInvalidSchema, check_schema

__all__ = [
    'make_assistant_tool_call_blocks',
    'make_tool_def',
    'make_named_tool_choice',
    'make_named_json_schema',
    'convert_from_constraints',
    'make_callable_end',
]

if TYPE_CHECKING:
    from sandbox import Sandbox


def make_assistant_tool_call_blocks(
    end: EndGenCallableTypes,
) -> list[ChatCompletionMessageToolCallParam]:
    """
    Makes tool call blocks for an assistant message from a generated delta with a callable end.

    Args:
        end: The end to add.

    Returns:
        The tool calls.
    """
    return [
        ChatCompletionMessageToolCallParam(
            id=id,
            function=chat_completion_message_tool_call_param.Function(
                name=constraint.name,
                arguments=json.dumps(content) if not isinstance(content, str) else content,
            ),
            type='function',
        )
        for constraint, content, id in zip(end.constraints, end.content, end.ids)
    ]


async def make_tool_def(
    executable: 'Executable', sandbox: 'Sandbox', guided: bool
) -> ChatCompletionToolParam:
    """
    Makes a tool definition.

    Args:
        executable: The executable to add.
        guided: Whether the executable is guided.

    Returns:
        The tool definition.
    """
    schema = await executable.schema(sandbox) | {'additionalProperties': False}
    return ChatCompletionToolParam(
        type='function',
        function=FunctionDefinition(
            name=executable.name,
            description=executable.description,
            parameters=schema,
            strict=guided,
        ),
    )


def make_named_tool_choice(
    executable: 'Executable',
) -> ChatCompletionNamedToolChoiceParam:
    """
    Makes a named tool choice.

    Args:
        executable: The executable to add.

    Returns:
        The named tool choice.
    """
    return ChatCompletionNamedToolChoiceParam(
        type='function',
        function=Function(
            name=executable.name,
        ),
    )


async def make_named_json_schema(
    executables: list['Executable'], sandbox: 'Sandbox', guided: bool
) -> ResponseFormatJSONSchema:
    """
    Makes a named JSON schema param for forcing generation to output a JSON object.
    If there is more than one executable, the JSON schema is a union of the JSON schemas of the executables.

    Args:
        executables: The executables to add.
        guided: Whether the executables are guided.

    Returns:
        The named JSON schema.
    """
    if len(executables) > 1:
        schema = {'anyOf': [await e.schema(sandbox) for e in executables]} | {
            'additionalProperties': False
        }
        name = ' '.join(e.name for e in executables)
        description = 'Any of the following:\n' + '\n'.join(
            f'- {e.description}' for e in executables
        )
    elif len(executables) == 1:
        schema = await executables[0].schema(sandbox) | {'additionalProperties': False}
        name = executables[0].name
        description = executables[0].description
    else:
        raise ValueError("No executables provided")
    return ResponseFormatJSONSchema(
        type='json_schema',
        json_schema=JSONSchema(
            name=name,
            description=description,
            schema=schema,  # type: ignore
            strict=guided,
        ),
    )


async def convert_from_constraints(
    ctx: Context, constraints: list['Constraint'], validate: bool = True
) -> dict[str, Any]:
    """
    Makes a dictionary of kwargs according to a list of generation constraints for the OpenAI Chat Completions API.
    Includes:
    - stop tokens
    - response format
    - max tokens
    - tools
    - tool choice

    Args:
        ctx: The context.
        constraints: The constraints.

    Returns:
        The kwargs.
    """
    from com.context_json import CallableConstraint, ObjectConstraint

    kwargs = {}

    if ctx.gen.type != 'json' and ctx.gen.guided:
        raise ValueError(f"Support for code constraints is not implemented yet")

    stop_tokens = [c.token for c in constraints if isinstance(c, StopTokenConstraint)]
    objects = [c for c in constraints if isinstance(c, ObjectConstraint)]
    callables = [c for c in constraints if isinstance(c, CallableConstraint)]
    max_tokens = [c.max_tokens for c in constraints if isinstance(c, MaxTokensConstraint)]

    if stop_tokens:
        kwargs['stop'] = stop_tokens

    if objects and objects[0].executables:
        kwargs['response_format'] = await make_named_json_schema(
            objects[0].executables,
            ctx.sandbox,
            objects[0].guided,
        )

    if max_tokens:
        kwargs['max_completion_tokens'] = max_tokens[0]

    if callables:
        kwargs['tools'] = [
            await make_tool_def(c.executable, ctx.sandbox, c.guided) for c in callables
        ]

    if len(callables) == 1:
        kwargs['tool_choice'] = make_named_tool_choice(callables[0].executable)

    # Check if the schema is valid for OAI, and if not, disable constrained generation.
    if validate and 'tools' in kwargs:
        for tool in kwargs['tools']:
            try:
                if function := tool.get('function', None):
                    if schema := function.get('parameters', None):
                        check_schema(schema)
            except OpenAIInvalidSchema:
                tool['function']['strict'] = False

    return kwargs


# === Convert to ===


def make_callable_end(
    tool_calls: list[dict[str, Any]], constraints: list['CallableConstraint']
) -> EndGenCallableTypes:
    """
    Makes a GenEndCallableTypes object from a list of ChatCompletionMessageToolCall's in dict form and a list of callable constraints.

    Args:
        tool_calls: The tool calls.
        constraints: The callable constraints.

    Returns:
        The callable end.
    """
    from com.context_json import EndGenCallableTypes

    names, ids, contents = [], [], []
    if not tool_calls:
        raise ValueError("No tool calls provided")
    for tool_call in tool_calls:
        ty = tool_call['type']
        key = 'arguments' if ty == 'function' else 'input'
        names.append(tool_call[ty]['name'])
        ids.append(tool_call['id'])
        contents.append(tool_call[ty][key])
    cons = [next((c for c in constraints if c.name == name), None) for name in names]
    if None in cons:
        missing = [name for name, c in zip(names, cons) if c is None]
        raise ValueError(f"Some constraints were not found: {missing}")
    return EndGenCallableTypes(
        constraints=list(cons),
        ids=ids,
        content=contents,
    )
