from pathlib import Path
from types import FunctionType, ModuleType

from agentica_internal.core.mixin import bases, init_module
from jinja2 import Environment

__all__ = [
    '__template_vars__',
    '__template_var_map__',
    'jinja_env',
    'JINJA_ENV_CACHE',
]


JINJA_ENV_CACHE: dict[str, 'Environment'] = {}

_BASE_DIR = Path(__file__).parent
_REPL_TXT_DIR = _BASE_DIR / 'repl_tool' / 'multi_turn' / 'text'

_TEMPLATE_VAR_MAP: dict[str, str] = {}


@init_module
def init(mod: ModuleType) -> None:
    var_map: dict[str, str] = {}

    # inherit from base modules
    for base in bases(mod):
        var_map |= base.__template_var_map__()

    # scan current module for get_*/is_*/has_* functions
    for name, value in mod.__dict__.items():
        if isinstance(value, FunctionType):
            if name.startswith('get_'):
                var_map[name.removeprefix('get_')] = name
            elif name.startswith(('is_', 'has_')):
                var_map[name] = name

    mod._TEMPLATE_VAR_MAP = var_map  # type: ignore[attr-defined]


def __template_var_map__() -> dict[str, str]:
    return _TEMPLATE_VAR_MAP


def __template_vars__() -> set[str]:
    return set(__template_var_map__())


def jinja_env(*template_paths: 'Path') -> 'Environment':
    from jinja2 import Environment, FileSystemLoader

    key = ':'.join(p.as_posix() for p in template_paths)
    if env := JINJA_ENV_CACHE.get(key):
        return env

    # security notice: yes, we use jinja directly, this is not a web application so we do not have flask.
    # templates and all variables are completely controlled by *us*, no user input makes it through.
    # jinja is used to format our agent prompts.
    searchpath = list(template_paths) + [_REPL_TXT_DIR]
    JINJA_ENV_CACHE[key] = env = Environment(
        loader=FileSystemLoader(searchpath=searchpath),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add include as a callable so we can use {{ include('/path.txt') }}
    def include_func(template_name: str) -> str:
        """Include a template and return its rendered content as a string."""
        template_name = template_name.lstrip('/')
        template = env.get_template(template_name)
        return template.render().strip()

    env.globals['include'] = include_func

    return env
