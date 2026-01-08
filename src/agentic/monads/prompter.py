import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from agentica_internal.core.mixin import mixin, module
from agentica_internal.session_manager_messages import PromptTemplate, should_template
from jinja2 import Environment, meta

from agentic.monads.safe_formatter import SafeFormatter
from com.abstract import HistoryMonad
from com.do import do
from com.monads import *

from . import template
from .template import *

__all__ = [
    '_format_custom_prompt',
    '_formatter',
    '_PROMPT_FORMATTER',
    '_user_prompt',
    '_system_prompt',
    '_prompt_from_file',
    '_template_jinja_env',
    '_jinja_search_paths',
    '_get_all_template_variables',
    '__template_vars__',
    '__template_var_map__',
    'jinja_env',
]

# no-op formatter in this base class
_PROMPT_FORMATTER: SafeFormatter = SafeFormatter()

type OptNames = list[str] | None

# inherit from template module
mixin(template)


def _get_all_template_variables(
    env: Environment, template_name: str, visited: set[str] | None = None
) -> set[str]:
    """Recursively find all undeclared variables in a template and its includes."""

    return get_all_template_variables(env, template_name, visited)


def _jinja_search_paths(template_dir: Path) -> list[Path]:
    """Return the Jinja template search paths for the given directory."""
    return [template_dir, template_dir.parent]


def _template_jinja_env(template_dir: Path) -> 'Environment':
    """Get a Jinja environment with the module's search paths."""
    search_paths = _jinja_search_paths(template_dir)
    return jinja_env(*search_paths)


@do(HistoryMonad[str])
def _prompt_from_file(
    template_dir: Path,
    file_name: str,
    task: str | None = None,
    premise: str | None = None,
    system: str | None = None,
    **kwargs: dict[str, Any],
):
    env = _template_jinja_env(template_dir)
    all_required_vars = _get_all_template_variables(env, file_name)
    all_required_values = {}

    # we populate this from args
    args_vars = {}
    if isinstance(task, str):
        args_vars['task'] = task
    if isinstance(premise, str):
        args_vars['premise'] = premise
    if isinstance(system, str):
        args_vars['system'] = system

    mod = module()
    mod_vars = __template_vars__()
    mod_var_map = __template_var_map__()

    session_info: ReplSessionInfo = yield repl_session_info()
    session_vars = session_info.__template_vars__()

    for var in all_required_vars:
        if var.startswith('_'):  # for loops
            continue
        if var in args_vars:
            value = args_vars[var]

        elif var in session_vars:
            value = getattr(session_info, var)

        elif var in mod_vars:
            func = getattr(mod, mod_var_map[var])
            func_opts = kwargs.get(var, {})
            monad = func(**func_opts)
            assert isinstance(monad, HistoryMonad)
            value = yield monad
        else:
            raise missing_var_error(var, set(args_vars), mod_vars, session_vars)

        all_required_values[var] = value.strip() if isinstance(value, str) else value

        if var == 'tool_descriptions':
            all_required_values[var] = [
                v if v is not None else '' for v in all_required_values[var].values()
            ]

        if var == 'tool_schemas':
            all_required_values[var] = [
                json.dumps(v, indent=2) for v in all_required_values[var].values()
            ]

    # security notice: all templates AND variables are completely controlled by *us*, no user input makes it through.
    template = env.get_template(file_name)
    rendered = template.render(**all_required_values).strip()
    rendered = dedent(rendered).strip()
    yield pure(rendered)


@do(HistoryMonad[str])
def _system_prompt(
    template_dir: Path,
    premise: str | None = None,
    system: str | PromptTemplate | None = None,
    **kwargs: dict[str, Any],
):
    if system_template := should_template(system):
        session: ReplSessionInfo = yield repl_session_info()

        # Agentic function gets RETURN_TYPE and STUBS injected into the system prompt.
        stubs = session.global_resources_stub
        extra = {'STUBS': stubs}
        if session.is_function:
            extra['RETURN_TYPE'] = session.return_type

        formatted = yield _formatter(system_template, extra)
        formatted = dedent(formatted).strip()
        yield pure(formatted)
    elif system and isinstance(system, str):
        formatted = dedent(system).strip()
        yield pure(formatted)
    else:
        session: ReplSessionInfo = yield repl_session_info()

        if premise is None:
            premise = ""
        sub_dir = "function" if session.is_function else "agent"
        prompt = yield _prompt_from_file(
            template_dir / sub_dir, "system.txt", premise=premise, **kwargs
        )
        prompt = dedent(prompt).strip()
        yield pure(prompt)


@do(HistoryMonad[str])
def _user_prompt(
    template_dir: Path,
    task: str | PromptTemplate | None = None,
    premise: str | None = None,
    system: str | PromptTemplate | None = None,
    **kwargs: str,
):
    session: ReplSessionInfo = yield repl_session_info()

    if isinstance(task, PromptTemplate):
        # Custom user prompt:
        # - RETURN_TYPE: the return type of the function
        # - STUBS: formatted python stubs
        # - USER_PROMPT: the normal prompt formatting without the task
        stubs = session.local_resources_stub
        return_type = session.return_type
        prompt = yield _prompt_from_file(
            template_dir / session.role,
            "user.txt",
            task="",
            premise=premise,
            system=system,
            **kwargs,
        )
        prompt = dedent(prompt).strip()
        formatted = yield _formatter(
            task.template,
            {
                'RETURN_TYPE': return_type,
                'STUBS': stubs,
                'USER_PROMPT': prompt,
            },
        )
        formatted = dedent(formatted).strip()
        yield pure(formatted)
    elif system and isinstance(system, str) and task:
        formatted = dedent(task).strip()
        yield pure(formatted)
    else:
        if premise is None:
            premise = ""
        if isinstance(task, PromptTemplate):
            task = task.template
        prompt = yield _prompt_from_file(
            template_dir / session.role,
            "user.txt",
            task=task,
            premise=premise,
            system=None,
            **kwargs,
        )
        prompt = dedent(prompt).strip()
        yield pure(prompt)


@do(HistoryMonad[str])
def _formatter(
    prompt: str,
    kwargs: dict[str, str] | None = None,
):
    yield pure(_format_custom_prompt(prompt, kwargs or {}))


def _format_custom_prompt(
    prompt: str,
    kwargs: dict[str, str] | None = None,
) -> str:
    return _PROMPT_FORMATTER.format(prompt, kwargs or {})


def missing_var_error(
    var: str, args_vars: set[str], mod_vars: set[str], repl_vars: set[str]
) -> Exception:
    line1 = f"Cannot fill template variable {var!r}\n"
    line2 = f"Variables from template call: {cat(args_vars)}\n"
    line3 = f"Variables from template class: {cat(mod_vars)}\n"
    line4 = f"Variables from repl session info: {cat(repl_vars)}"
    raise PromptTemplateError(line1 + line2 + line3 + line4)


cat = ' '.join


class PromptTemplateError(Exception):
    pass


def get_all_template_variables(
    env: Environment, template_name: str, visited: set[str] | None = None
) -> set[str]:
    """
    Recursively find all undeclared variables in a template and its includes.

    The results are cached on the jinja environment itself.
    """

    cache = getattr(env, "__all_template_vars_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "__all_template_vars_cache", cache)

    if template_name in cache:
        return cache[template_name]

    if visited is None:
        visited = set()

    # Avoid infinite recursion from circular includes
    if template_name in visited:
        return set()
    visited.add(template_name)

    # Get template source and parse it

    template_source = env.loader.get_source(env, template_name)[0]
    ast = env.parse(template_source)

    # Find undeclared variables in this template
    undeclared_vars = meta.find_undeclared_variables(ast)

    # Find all included templates
    included_templates = set()
    for node in ast.find_all((meta.nodes.Include,)):
        if hasattr(node, 'template') and hasattr(node.template, 'value'):
            # Static include like {% include 'function/starter.txt' %}
            included_templates.add(node.template.value)

    # Recursively get variables from included templates
    for included in included_templates:
        undeclared_vars |= get_all_template_variables(env, included, visited)

    cache[template_name] = undeclared_vars

    return undeclared_vars
