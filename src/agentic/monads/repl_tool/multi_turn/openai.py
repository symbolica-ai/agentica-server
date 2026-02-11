from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import yaml
from agentica_internal.core.mixin import mixin

from agentic.monads.common import REPL_TXT_DIR, text_not_between
from com.abstract import HistoryMonad
from com.do import do
from com.monads import *
from inference.endpoint import Generation

from ... import prompter
from ...prompter import *

__all__ = [
    '_user_execution',
    '_user_instructions',
    'monad',
    'user_monad',
    '_template_dir',
    'system_monad',
    'interaction_monad',
    '_explain',
    '_formatter',
    '_prompt_from_file_no_vars',
    '_format_custom_prompt',
    '_user_prompt',
    '_system_prompt',
]

# inherit from the prompter base-module
mixin(prompter)


if TYPE_CHECKING:
    from agentica_internal.session_manager_messages import PromptTemplate


type PromptType = 'str | PromptTemplate'


def _user_execution(text: str):
    return insert_string(text, ('user', 'execution'))


def _user_instructions(text: str):
    return insert_string(text, ('user', 'instructions'))


@do(HistoryMonad[None])
def monad(user_prompt: str, initial_call: bool = True):
    if initial_call:
        yield system_monad()

    # Insert the provided user instructions.
    yield user_monad(user_prompt)

    # Agent goes brrrrr...
    yield interaction_monad()

    yield pure(None)


@do(HistoryMonad[None])
def user_monad(
    user_prompt: PromptType,
    system: PromptType | None = None,
):
    prompt = yield _user_prompt(_template_dir(), task=user_prompt, system=system)
    yield _user_instructions(prompt)


def _template_dir():
    return REPL_TXT_DIR / "openai"


@do(HistoryMonad[None])
def system_monad(
    premise: str | None = None,
    system: PromptType | None = None,
):
    session: ReplSessionInfo = yield repl_session_info()
    template_dir = _template_dir()
    prompt = yield _system_prompt(template_dir, premise=premise, system=system)
    yield insert_string(role='system', content=prompt)

    if system is not None or session.is_function:
        return

    yield _few_shot_examples()


@do(HistoryMonad[None])
def _few_shot_examples():
    """Insert few-shot examples from explain/few-shot.yaml into the conversation.

    Supports two output formats based on session.uses_tool_calls:
    - Tool call mode: assistant code is emitted as function_call items
    - Markdown mode: assistant code is embedded in ```python blocks

    YAML format supports:
    - `text`: reasoning/explanation (optional)
    - `code`: Python code to execute (optional)
    - `value`: backward-compat markdown format or dict for instructions
    """
    session: ReplSessionInfo = yield repl_session_info()

    # Load the user.txt template for rendering instructions
    template_dir = _template_dir() / session.role
    env = _template_jinja_env(template_dir)
    user_template = env.get_template("user.txt")

    # Load few-shot examples from YAML (render through Jinja first for includes)
    few_shot_file = REPL_TXT_DIR / "explain" / "few-shot.yaml"
    if not few_shot_file.exists():
        return

    few_shot_raw = few_shot_file.read_text()
    few_shot_rendered = env.from_string(few_shot_raw).render(session.__dict__)
    examples = yaml.safe_load(few_shot_rendered)

    for example in examples:
        role = example["role"]

        if role == 'instructions':
            # Render the user template with the provided variables
            value = example["value"]
            assert isinstance(value, dict)
            # security notice: all templates AND variables are completely controlled by *us*
            rendered = user_template.render(**value)
            yield _user_instructions(dedent(rendered).strip())

        elif role == 'assistant':
            text = example.get("text", "")
            code = example.get("code")
            value = example.get("value")  # backward-compat markdown format
            final = example.get("final", False)

            if session.uses_tool_calls:
                if code:
                    # Emit as function_call
                    yield insert_function_call("python", dedent(code).strip(), text)
                    if final:
                        yield insert_execution_result("[Result returned to user]")
                elif text and not value:
                    # Text-only assistant turn (no code) - emit as regular message
                    # First close any pending tool call
                    yield insert_string(role='assistant', content=dedent(text).strip())
                elif value:
                    # Backward-compat: value contains markdown with multiple code blocks
                    # Skip this in tool call mode (multiple code blocks don't apply)
                    # Also skip the pending call_id tracking since we're skipping this turn
                    pass
            else:
                # Markdown mode
                if value:
                    # Use value directly (backward compat)
                    content = dedent(value).strip()
                else:
                    # Construct markdown from text + code
                    content = dedent(text).strip() if text else ""
                    if code:
                        code_block = f"```python\n{dedent(code).strip()}\n```"
                        content = f"{content}\n{code_block}" if content else code_block
                if content:
                    yield insert_string(role='assistant', content=content)

        elif role == 'execution':
            value = dedent(example["value"]).strip()
            if session.uses_tool_calls:
                # Emit as function_call_output
                yield insert_execution_result(value)
            elif not session.uses_tool_calls:
                # Emit as user message (markdown mode)
                yield _user_execution(value)
            # If uses_tool_calls but no pending_call_id, skip (orphan execution)

        else:
            raise ValueError(f"Invalid role: {role}")


@do(HistoryMonad[None])
def interaction_monad():
    session: ReplSessionInfo = yield repl_session_info()

    # Let the agent execute some code and gain feedback in a loop.
    while True:
        response: Generation = yield model_inference()

        # Code is extracted by InferenceSystem (from tool call or markdown)
        code_block: str | None = response.code

        # If no content AND no code, show empty-response error.
        # Note: When using tool calls, output_text may be empty but code is present.
        if not response.content and not code_block:
            msg = yield _explain("empty-response.txt")
            yield insert_execution_result(msg)
            continue

        # Special case: if returning text and no code found, treat clean response as return
        if code_block is None and session.is_returning_text:
            *_, content = text_not_between(response.content, "<thinking>", "</thinking>")
            *_, content = text_not_between(
                content, "<implementation_analysis>", "</implementation_analysis>"
            )
            if content := content.strip():
                code_block = f"return {content!r}"

        if code_block is None:
            msg = yield _explain("missing-code.txt")
            yield insert_execution_result(msg)
            continue

        exec_id = yield log_code_block(code_block)
        summary: ReplEvaluationInfo = yield repl_run_code(code_block)
        output: str = summary.output
        yield log_execute_result(summary.output, exec_id)

        # a FutureResultMsg has been sent - submit output then return
        if summary.has_result:
            # When using tool calls, we MUST provide function_call_output for the tool call.
            # Otherwise, the next turn will fail with "No tool output found for function call".
            if session.uses_tool_calls:
                if output and not output.isspace():
                    yield insert_execution_result(output + "\n")
                else:
                    # Submit a completion marker when there's no stdout output
                    yield insert_execution_result("[Execution completed]\n")
            return

        # Submit execution output for non-returning code
        if output and not output.isspace():
            yield insert_execution_result(output + "\n")
        else:
            # if no repl output provided, provide guidance
            msg = yield _explain("empty-output.txt")
            yield insert_execution_result(msg)

        # if repl raised a SystemExit, provide guidance
        if summary.exception_name == 'SystemExit':
            msg = yield _explain("uncaught-exit.txt")
            yield insert_execution_result(msg)

        # if there were more code blocks, tell agent we didn't run them
        if response.extra_code_blocks > 0:
            assert not response.code_from_tool, (
                "Parallel tool calling is set to False, inference endpoint should never return extra code blocks."
            )

            msg = yield _explain("multiple-code-blocks.txt")
            yield insert_execution_result(msg)

    yield pure()


@do(HistoryMonad[str])
def _explain(template_name: str):
    """Load an explanation template from the explain/ directory and render it with session vars."""
    template_dir = REPL_TXT_DIR / "explain"
    yield _prompt_from_file_no_vars(template_dir, template_name)


@do(HistoryMonad[str])
def _formatter(
    prompt: str,
    kwargs: dict[str, str] | None = None,
):
    # Render Jinja templates

    session_info: ReplSessionInfo = yield repl_session_info()

    base_dir = _template_dir()
    sub_dir = base_dir / session_info.role

    interactions = yield _prompt_from_file_no_vars(sub_dir, "interactions.txt")
    notes = yield _prompt_from_file_no_vars(sub_dir, "notes.txt")
    objectives = yield _prompt_from_file_no_vars(sub_dir, "objectives.txt")
    output = yield _prompt_from_file_no_vars(sub_dir, "output.txt")
    starter = yield _prompt_from_file_no_vars(sub_dir, "starter.txt")
    workflow = yield _prompt_from_file_no_vars(sub_dir, "workflow.txt")

    kwargs = {
        'INTERACTIONS': interactions,
        'NOTES': notes,
        'OBJECTIVES': objectives,
        'OUTPUT': output,
        'STARTER': starter,
        'WORKFLOW': workflow,
        **(kwargs or {}),
    }
    formatted = _format_custom_prompt(prompt, kwargs)
    yield pure(formatted)


def _prompt_from_file_no_vars(template_dir: Path, file_name: str):
    return _prompt_from_file(template_dir, file_name, task='', premise='', system='')
