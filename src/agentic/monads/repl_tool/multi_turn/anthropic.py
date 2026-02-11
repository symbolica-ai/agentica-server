from pathlib import Path
from textwrap import indent

from agentica_internal.core.mixin import mixin

from agentic.monads.common import REPL_TXT_DIR
from agentic.monads.safe_formatter import SafeFormatter
from com.abstract import HistoryMonad
from com.do import do
from com.monads import *

from . import openai
from .openai import *

__all__ = [
    '_user_execution',
    '_user_instructions',
    'monad',
    'user_monad',
    '_template_dir',
    '_jinja_search_paths',
    'system_monad',
    'interaction_monad',
    '_formatter',
    '_prompt_from_file_no_vars',
    '_format_custom_prompt',
    '_user_prompt',
    '_system_prompt',
]

# anthropic prompting inherits from openai prompting:
# - change the user instruction and execution messages to not use usernames
# - change system prompt formatting to use anthropic optimized prompts
mixin(openai)


def _user_execution(text: str):
    return insert_string('<execution>\n' + text.rstrip() + '\n</execution>', ('user', None))


@do(HistoryMonad[None])
def _user_instructions(text: str):
    session: ReplSessionInfo = yield repl_session_info()
    is_agentic_function = session.is_function
    if is_agentic_function:
        yield insert_string(text, ('user', None))
    else:
        yield insert_string(
            '<instructions>\n' + indent(text.strip(), ' ' * 2) + '\n</instructions>', ('user', None)
        )


_PROMPT_FORMATTER: SafeFormatter = SafeFormatter()


def _template_dir():
    return REPL_TXT_DIR / "anthropic"


def _jinja_search_paths(template_dir: Path) -> list[Path]:
    """Return the Jinja template search paths for anthropic (with openai as fallback)."""
    role = template_dir.name  # e.g., "agent" or "function"
    openai_dir = openai._template_dir()
    # [anthropic/role, anthropic, openai/role, openai]
    return [template_dir, template_dir.parent, openai_dir / role, openai_dir]
