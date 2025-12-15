from pathlib import Path
from textwrap import dedent

from agentica_internal.session_manager_messages import PromptTemplate

from com.abstract import HistoryMonad
from com.do import do
from com.monads import *

from ..base import Prompter
from ..common import TemplateClass, text_not_between

__all__ = [
    'MultiTurnJSON',
]


class MultiTurnJSON(TemplateClass):
    @classmethod
    @do(HistoryMonad[None])
    def interaction_monad(cls):
        session: ReplSessionInfo = yield repl_session_info()

        while True:
            response = yield gen()
            yield update_with_execute(response)
            evaluation: ReplEvaluationInfo = yield repl_

            # Check if generation stopped not due to an executable and if the return type is str, we're done.
            is_str = yield execute("__return_type == str")
            continuing = yield end_executed()
            if is_str == "True" and response.content and continuing:
                *_, content = text_not_between(response.content, "<thinking>", "</thinking>")
                *_, content = text_not_between(
                    content, "<implementation_analysis>", "</implementation_analysis>"
                )
                yield execute(f'result = {repr(content.strip())}')

            # If result was successfully set, we're done.
            finished = yield has_local_var('result')

            if finished:
                yield return_var('result')
                break

    @classmethod
    @do(HistoryMonad[None])
    def system_monad(
        cls,
        premise: str | None = None,
        system: str | PromptTemplate | None = None,
    ):
        yield cls.add_agent_error_tool()
        is_agentic_fn = yield is_agentic_function()
        whitelist, blacklist = [], []
        if is_agentic_fn:
            yield cls.add_return_tool()
        else:
            whitelist.append('agent_error_tool')

        is_str = yield execute("__is_returning_text")
        if is_str == "True":
            blacklist.append('return_tool')

        kwargs = {
            k: {"blacklist": blacklist} for k in ['tool_names', 'tool_schemas', 'tool_descriptions']
        }
        if whitelist:
            kwargs.update(
                {
                    k: {"whitelist": whitelist, "blacklist": kwargs[k]['blacklist']}
                    for k in ['tool_names', 'tool_schemas', 'tool_descriptions']
                }
            )

        template_dir = Path(__file__).parent / "text" / "openai"
        prompt = yield Prompter._system_prompt(
            template_dir,
            premise=premise,
            system=system,
            **kwargs,
        )
        yield insert_string(prompt, name='system')

    @classmethod
    @do(HistoryMonad[None])
    def user_monad(
        cls, user_prompt: str | PromptTemplate, system: str | PromptTemplate | None = None
    ):
        is_agentic_fn = yield is_agentic_function()

        blacklist = []

        if not is_agentic_fn:
            yield cls.add_return_tool()
            yield cls.add_exception_tools()
            blacklist.append('agent_error_tool')

        is_str = yield execute("__is_returning_text")
        if is_str == "True":
            blacklist.append('return_tool')

        kwargs = {
            k: {"blacklist": blacklist} for k in ['tool_names', 'tool_schemas', 'tool_descriptions']
        }

        template_dir = Path(__file__).parent / "text" / "standard"
        prompt = yield Prompter._user_prompt(
            template_dir, task=user_prompt, system=system, **kwargs
        )
        yield insert_string(prompt, name='user')

    @classmethod
    @do(HistoryMonad[str])
    def add_return_tool(cls):
        return_tool = dedent('''
        def return_tool(response: __return_type) -> __return_type:
            """Return the value back to the user."""
            global result
            result = response
            return response

        _hide_variable('return_tool')
        ''').strip()

        yield execute(return_tool, snippet=True)

        yield add_executable(name='return_tool', type='callable')

        yield pure('return_tool')

    @classmethod
    @do(HistoryMonad[str])
    def add_agent_error_tool(cls):
        agent_error_tool = dedent('''
        def agent_error_tool(message: str) -> None:
            """Raise an error."""
            raise AgentError(RuntimeError(message))

        _hide_variable('agent_error_tool')
        ''').strip()

        yield execute(agent_error_tool, snippet=True)

        yield add_executable(name='agent_error_tool', type='callable')

        yield pure('agent_error_tool')

    @classmethod
    @do(HistoryMonad[None])
    def add_exception_tools(cls):
        # Get schemas and descriptions of all objects
        names = yield get_executable_names(constraint_type='object')

        # Check if they subclass Exception
        for n in names:
            is_exception = (
                yield execute(f"isinstance({n}, type) and issubclass({n}, BaseException)")
            ).strip() == "True"
            if not is_exception:
                continue
            exception_tool = f'''
            def raise_{n}(message: str) -> None:
                """Raise a {n} exception."""
                raise AgentError({n}(message))
            '''.strip()
            yield execute(exception_tool)
            yield add_executable(name=f"raise_{n}", type='callable')
