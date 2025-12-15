from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import yaml
from agentica_internal.core.mixin import mixin

from agentic.monads.common import REPL_TXT_DIR, text_between, text_not_between
from agentic.monads.safe_formatter import SafeFormatter
from com.abstract import HistoryMonad
from com.deltas import *
from com.do import do
from com.monads import *
from com.roles import AgentRole, SystemRole, UserRole

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
    return insert_string(text, name=UserRole("execution"))


def _user_instructions(text: str):
    return insert_string(text, name=UserRole("instructions"))


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
    yield insert_string(prompt, name=SystemRole())

    if system is not None or session.is_function:
        return

    yield _few_shot_examples()


@do(HistoryMonad[None])
def _few_shot_examples():
    """Insert few-shot examples from explain/few-shot.yaml into the conversation."""
    session: ReplSessionInfo = yield repl_session_info()

    # Load the user.txt template for rendering instructions
    template_dir = _template_dir() / session.role
    env = _template_jinja_env(template_dir)
    user_template = env.get_template("user.txt")

    # Collect available template variables
    session_vars = session.__template_vars__()
    user_vars = _get_all_template_variables(env, "user.txt")
    available_vars = session_vars | user_vars

    # Load few-shot examples from YAML (render through Jinja first for includes)
    few_shot_file = REPL_TXT_DIR / "explain" / "few-shot.yaml"
    if not few_shot_file.exists():
        return

    few_shot_raw = few_shot_file.read_text()
    few_shot_rendered = env.from_string(few_shot_raw).render()
    examples = yaml.safe_load(few_shot_rendered)

    for example in examples:
        role = example["role"]
        value = example["value"]

        if role == 'instructions':
            # Render the user template with the provided variables
            assert isinstance(value, dict)
            # security notice: all templates AND variables are completely controlled by *us*
            value = user_template.render(**value)

        value = dedent(value).strip()

        if role == 'assistant':
            yield insert_string(value, name=AgentRole())
        elif role == 'instructions':
            yield _user_instructions(value)
        elif role == 'execution':
            yield _user_execution(value)
        else:
            raise ValueError(f"Invalid role: {role}")


@do(HistoryMonad[None])
def interaction_monad():
    user_execution = _user_execution

    session: ReplSessionInfo = yield repl_session_info()

    # Let the agent execute some code and gain feedback in a loop.
    while True:
        response: GeneratedDelta = yield model_inference()
        yield insert_delta(response)

        if not response.content:
            msg = yield _explain("empty-response.txt")
            yield user_execution(msg)
            continue

        code_blocks = list(text_between(response.content, "```python", "```"))
        if not code_blocks and session.is_returning_text:
            # if agent provided clean response and the return type is a string,
            # treat it as an attempt to return the string
            *_, content = text_not_between(response.content, "<thinking>", "</thinking>")
            *_, content = text_not_between(
                content, "<implementation_analysis>", "</implementation_analysis>"
            )
            if content := content.strip():
                code_blocks = [f"return {content!r}"]

        if not code_blocks:
            msg = yield _explain("missing-code.txt")
            yield user_execution(msg)
            continue

        code_block, *extra = code_blocks

        exec_id = yield log_code_block(code_block)
        summary: ReplEvaluationInfo = yield repl_run_code(code_block)
        output: str = summary.output
        yield log_execute_result(summary.output, exec_id)

        # a FutureResultMsg has been sent
        if summary.has_result:
            return

        # if no repl output provided, provide guidance
        if not output or output.isspace():
            msg = yield _explain("empty-output.txt")
            yield user_execution(msg)
        else:
            yield user_execution(output + "\n")

        # if no repl raised a SystemExit, provide guidance
        if summary.exception_name == 'SystemExit':
            msg = yield _explain("uncaught-exit.txt")
            yield user_execution(msg)

        # if there were more code blocks, tell agent we didn't run them.
        if extra:
            msg = yield _explain("multiple-code-blocks.txt")
            yield user_execution(msg)

    yield pure()


@do(HistoryMonad[str])
def _explain(template_name: str):
    """Load an explanation template from the explain/ directory and render it with session vars."""
    template_dir = REPL_TXT_DIR / "explain"
    yield _prompt_from_file_no_vars(template_dir, template_name)


_PROMPT_FORMATTER: SafeFormatter = SafeFormatter()


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
