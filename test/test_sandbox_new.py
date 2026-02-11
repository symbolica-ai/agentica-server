import pytest

from sandbox import Sandbox
from sm_testing import SandboxTestContext


@pytest.mark.asyncio
async def test_sandbox_repl():
    sandbox = Sandbox.testing_sandbox()
    info = await sandbox.repl_run_code('print(5)')
    assert info.output == '5\n'
    assert not info.has_result

    info = await sandbox.repl_run_code('raise KeyError(1)')
    assert info.exception_name == 'builtins.KeyError'

    info = await sandbox.repl_run_code('return 5')
    assert info.has_return_value

    await sandbox.repl_call_method('set_return_type', str)
    info = await sandbox.repl_run_code('return 5')
    assert not info.has_return_value
    assert (
        info.traceback_str
        == '''
Traceback (most recent call last):
  <repl> line 1
    return 5
    
typeguard.TypeCheckError: int is not an instance of str
cannot return value: expected str, got int
'''.lstrip()
    )

    info = await sandbox.repl_run_code("return 'foo'")
    assert info.has_return_value

    out, stdout, stderr = await sandbox.repl_command('1 + 2')
    assert out == '3'
    assert stdout == '3'
    assert stderr == ''

    info = await sandbox.repl_run_code('def my_function(s): return len(s)')
    out, stdout, stderr = await sandbox.repl_eval('my_function("abc")')
    assert out == '3'
    assert stdout == ''
    assert stderr == ''

    await sandbox.aclose()


def foo():
    return 99


@pytest.mark.asyncio
async def test_sandbox_context_new():
    async with SandboxTestContext('test_sandbox_context_new', foo) as sandbox:
        info = await sandbox.repl_run_code('foo()')
        assert info.out_str == '99'

        lst = await sandbox.repl_dir_vars('locals')
        assert lst == []

        lst = await sandbox.repl_dir_vars('globals')
        assert 'foo' in lst

        lst = await sandbox.repl_dir_vars('user')
        assert 'foo' in lst

        var_info = await sandbox.repl_var_info('foo')
        assert var_info is not None
        assert not var_info.is_class
        assert var_info.is_callable
        assert var_info.safe_repr == 'foo'

        await sandbox.repl_run_code('atom = 512')
        var_info = await sandbox.repl_var_info('atom')
        assert var_info is not None
        assert not var_info.is_class
        assert not var_info.is_callable
        assert var_info.safe_repr == '512'

        await sandbox.repl_run_code('anno = list[str]')
        var_info = await sandbox.repl_var_info('anno')
        assert var_info is not None
        assert not var_info.is_class
        assert var_info.is_type_anno
        assert var_info.safe_repr == 'list[str]'

        await sandbox.repl_run_code('cls = str')
        var_info = await sandbox.repl_var_info('cls')
        assert var_info is not None
        assert var_info.is_class
        assert not var_info.is_type_anno
        assert var_info.safe_repr == 'str'
