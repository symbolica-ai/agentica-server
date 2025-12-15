# fmt: off

from typing import Any

from agentica_internal.repl import REPL_VAR

from sandbox.guest.agent_repl import AgentRepl


def foo(i: int) -> str:
    """foo doc"""
    return 'x' * i


FOO_STUB = 'def foo(i: int) -> str:\n    """foo doc"""'


def test_agent_repl_globals_locals():
    repl = AgentRepl()
    repl.initialize(
        local_vars=dict(x=5), global_vars=dict(foo=foo, hideme=99), hidden_vars=('hideme',)
    )
    assert repl.eval_expr('dir()') == ['x', 'foo']
    assert repl.eval_expr('list(globals().keys())') == ['foo', 'hideme']
    assert repl.eval_expr('list(locals().keys())') == ['x']


def test_agent_repl_recursion():
    repl = AgentRepl()
    info = repl.run_code('''
        def inf_loop():
            inf_loop()
        inf_loop()
    ''')
    assert info.error.traceback.strip() == '''
Traceback (most recent call last):
  <repl> line 4
    inf_loop()
  <repl> line 3, in inf_loop
    inf_loop()
  <repl> line 3, in inf_loop
    inf_loop()
  <repl> line 3, in inf_loop
    inf_loop()
  [Previous line repeated 12 more times]
RecursionError: maximum recursion depth exceeded
'''.strip()


def test_agent_repl_loaded_modules():
    repl = AgentRepl()
    assert repl.get_loaded_modules() == ()
    assert repl.run_code('import math')
    assert repl.get_loaded_modules() == ('math',)


def test_agent_repl_self_stubs():
    def agentic_fn(i: int, l: list[int]) -> str:
        """magic doc"""

    repl = AgentRepl()
    repl.initialize(
        local_vars=dict(),
        global_vars=dict(__self=agentic_fn, __role='function'),
    )

    info = repl.info
    assert info.is_function
    assert info.function_name == 'agentic_fn'
    assert info.function_argument_signature == '(i: int, l: list[int])'
    assert info.function_argument_names == ['i', 'l']
    assert info.function_description == 'magic doc'

    expected_stub = 'def agentic_fn(i: int, l: list[int]) -> str:\n    """magic doc"""'
    assert info.function_stub == expected_stub

    assert not info.has_global_resources
    assert info.global_resources_stub == ''

def strip_trailing_whitespace[T](s: T) -> T:
    """strip trailing whitespace on each line"""
    if isinstance(s, str):
        return '\n'.join(line.rstrip() for line in s.split('\n')) # type: ignore
    return s

def eq_multiline(a: Any, b: Any) -> bool:
    return strip_trailing_whitespace(a) == strip_trailing_whitespace(b)

def test_agent_repl_return_checks():
    repl = AgentRepl()
    repl.initialize(local_vars=dict(), global_vars=dict(__return_type=bool), hidden_vars=())

    summary = repl.run_code_info('return True')
    assert summary.has_return_value

    summary = repl.run_code_info('return 99')
    assert not summary.has_return_value

    expected_traceback_str = '''
Traceback (most recent call last):
  <repl> line 1
    return 99

typeguard.TypeCheckError: int is not an instance of bool
cannot return value: expected bool, got int
'''.lstrip()
    assert eq_multiline(summary.traceback_str, expected_traceback_str)


def test_agent_repl_return_type_stubs():
    repl = AgentRepl()
    repl.initialize(
        local_vars=dict(),
        global_vars=dict(__return_type=list[str]),
        hidden_vars=(),
    )
    info = repl.info
    assert info.return_type == 'list[str]'
    assert not info.is_returning_text

    repl.initialize(
        local_vars=dict(),
        global_vars=dict(__return_type=str),
        hidden_vars=(),
    )
    info = repl.info
    assert info.return_type == 'str'
    assert info.is_returning_text


def test_agent_repl_input_stubs():
    def magic_abc(a: int, b: str, c: bool) -> str:
        """magic doc"""

    repl = AgentRepl()
    repl.initialize(
        local_vars=dict(a=5, b='xxx', c=True),
        global_vars=dict(__self=magic_abc, __role='function', foo=foo),
        hidden_vars=(),
    )
    repl.run_code('z = 99')

    info = repl.info

    expected_input_reprs = {'a': '5', 'b': "'xxx'", 'c': 'True'}
    assert info.input_reprs == expected_input_reprs

    expected_inputs_stub = "a: int = 5\n\nb: str = 'xxx'\n\nc: bool = True"
    assert info.inputs_stub == expected_inputs_stub

    assert repl.info.global_resources_stub == FOO_STUB
    assert repl.info.globals.kinds == {'foo': 'function'}

    # test that global resources accumulate
    repl.initialize(global_vars={'bar': True})
    assert repl.info.global_resources_stub == f'bar: bool = True'
    assert repl.info.globals.kinds == {'foo': 'function', 'bar': 'data'}

    # test that local resources get reset
    repl.initialize(local_vars={'x': 'hello'})
    assert repl.info.locals.kinds == {'x': 'data'}
    assert repl.info.local_resources_stub == "x: str = 'hello'"

    repl.initialize(local_vars={'y': 'goodbye'})
    assert repl.info.locals.kinds == {'y': 'data'}
    assert repl.info.local_resources_stub == "y: str = 'goodbye'"


def test_agent_repl_long_string():
    repl = AgentRepl()
    long_str = '*' * 4096
    info = repl.initialize(global_vars={'long_str': long_str, REPL_VAR.MAX_REPR_LEN: 16})
    assert info['globals'].reprs.get('long_str') == '<str len=4096>'
