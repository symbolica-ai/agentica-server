from pathlib import Path

import pytest

from sm_testing import Sandbox, SandboxTestContext


@pytest.mark.asyncio
async def test_log_file(is_local_runner):
    tmp_file = Path('/tmp/test_log_file')
    tmp_file.unlink(missing_ok=True)
    sb = Sandbox.testing_sandbox(log_path=tmp_file, log_inherit=False, log_tags='AgentRepl+repl')
    await sb.repl_run_code('1 + 1')
    await sb.aclose()
    log_text = sb.read_log_file(ansi=False)
    assert "AgentRepl[" in log_text
    assert "AgentWorld[" not in log_text  # verify log_tags had an effect
    assert "<<< 1 + 1\n\n" in log_text


@pytest.mark.asyncio
async def test_warp(is_local_runner):
    customer_function_called = False

    def customer_function(a: str) -> int:
        nonlocal customer_function_called
        customer_function_called = True
        return len(a)

    test_ctx = SandboxTestContext('test_warp')
    test_ctx.add_globals(customer_function, my_global=5)
    test_ctx.add_locals(my_local=10)

    async with test_ctx as sandbox:
        local_names = await sandbox.repl_dir_vars('locals')
        assert local_names == ['my_local']
        global_names = await sandbox.repl_dir_vars('globals')
        assert set(global_names).issuperset({'customer_function', 'my_global'})
        result = await sandbox.repl_eval('customer_function("abc")')
        assert result == ('3', '', '')

        assert customer_function_called is True


@pytest.mark.asyncio
async def test_exec(make_dummy_sandbox, is_local_runner):
    fail_calls: list[str] = []
    exit_calls: int = 0

    async with make_dummy_sandbox(
        logging=False,
    ) as sb:
        # 1) Single-line expression shows value via displayhook
        out, stdout, stderr = await sb.repl_command("1 + 2")
        assert out == "3"
        assert stdout == "3"
        assert stderr == ""

        _, _, _ = await sb.repl_command("raise AgentError(ValueError('boom'))")
        last_exc = sb.eval_info.exception_name
        assert last_exc == 'builtins.ValueError'

        # assert "__last_exception" in set(await sb.repl_get_local_names())

        _, _, _ = await sb.repl_command("exit()")
        last_exc = sb.eval_info.exception_name
        assert last_exc == "builtins.SystemExit"

        parent_directory_of_this_file = Path(__file__).parent.resolve().as_posix()
        out, stdout, stderr = await sb.repl_command(
            f"import os; print(os.listdir('{parent_directory_of_this_file}'))"
        )
        if is_local_runner:
            assert "conftest.py" in out
            assert "conftest.py" in stdout
        else:
            # WASM runner should error when accessing host FS
            assert stdout == ""
            assert out.startswith('Traceback (most recent call last):\n  <repl> line 1')


@pytest.mark.asyncio
async def test_async_await(make_dummy_sandbox):
    async with make_dummy_sandbox(logging=False) as sb:
        code = """
async def foo():
    return 'OK'
async def main():
    return await foo()
value = await main()
print(value)
"""

        out, stdout, stderr = await sb.repl_exec(code)
        assert out == "OK\n"
        assert stdout == "OK\n"
        assert stderr == ""


@pytest.mark.asyncio
async def test_locals(make_dummy_sandbox):
    async with make_dummy_sandbox(logging=False) as sb:
        _ = await sb.repl_exec("def f(x: str) -> str: ...")

        xs = await sb.repl_dir_vars()
        assert 'f' in xs
