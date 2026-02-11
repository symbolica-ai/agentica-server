import asyncio
from dataclasses import dataclass
from textwrap import dedent

import pytest

from sm_testing import SandboxTestContext


@dataclass
class Foo:
    name: str
    rating: int


@dataclass
class Bar:
    value: float


class Baz: ...


def f_simple() -> None:
    """No args, no return"""
    raise NotImplementedError


def f_annotated(x: str, y: int = 2) -> str:
    """Function with defaults"""
    _ = (x, y)
    raise NotImplementedError


def f_varargs(x: int, *args: str, **kwargs: int) -> str:
    """Function with varargs and kwargs"""
    _ = (x, args, kwargs)
    raise NotImplementedError


async def f_async(a: float) -> float:
    """Async function"""
    _ = a
    raise NotImplementedError


def f_union(x: int | str) -> int | str:
    """Function with union return type"""
    _ = x
    raise NotImplementedError


def f_nested(x: dict[str, tuple[int, str]]) -> dict[dict[bool, int], float]:
    """Function with nested type"""
    _ = x
    raise NotImplementedError


def f_dc(a: Foo) -> Bar:
    """Convert Foo to Bar."""
    _ = a
    raise NotImplementedError


def f_builtins(u: set[int], v: list[str]) -> dict[str, int]:
    """Function using built-in generic types (set, list, dict)."""
    _ = (u, v)
    raise NotImplementedError


def f_container_dc(items: set[Foo]) -> list[Bar]:
    """Function with dataclass generics."""
    _ = items
    raise NotImplementedError


@pytest.mark.asyncio
async def test_function_stub_generation():
    async with SandboxTestContext(
        'test_function_stub_generation',
        Foo,
        Bar,
        Baz,
        f_simple,
        f_annotated,
        f_varargs,
        f_async,
        f_union,
        f_nested,
        f_dc,
        f_builtins,
        f_container_dc,
    ) as sb:
        names = {
            'Foo',
            'Bar',
            'Baz',
            'f_simple',
            'f_annotated',
            'f_varargs',
            'f_async',
            'f_union',
            'f_nested',
            'f_dc',
            'f_builtins',
            'f_container_dc',
        }
        local_names = await sb.repl_dir_vars('locals')
        global_names = await sb.repl_dir_vars('globals')
        assert set(global_names).issuperset(names)

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_simple)")
        assert stdout.startswith("def f_simple() -> None:")
        assert '    """No args, no return"""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_annotated)")
        # # print("f_annotated:\n", stdout)
        assert stdout.startswith("def f_annotated(x: str, y: int = 2) -> str:")
        assert '    """Function with defaults"""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_varargs)")
        # # print("f_varargs:\n", stdout)
        assert stdout.startswith("def f_varargs(x: int, *args: str, **kwargs: int) -> str:")
        assert '    """Function with varargs and kwargs"""' in stdout

        # Async functions are not supported yet
        _out, stdout, _stderr = await sb.repl_command("show_definition(f_async)")
        # print("f_async:\n", stdout)
        assert stdout.startswith("def f_async(a: float) -> Future[float]:")
        assert '    """Async function"""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_union)")
        # print("f_union:\n", stdout)
        assert stdout.startswith("def f_union(x: int | str) -> int | str:")
        assert '    """Function with union return type"""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_nested)")
        # print("f_nested:\n", stdout)
        assert stdout.startswith(
            "def f_nested(x: dict[str, tuple[int, str]]) -> dict[dict[bool, int], float]:"
        )
        assert '    """Function with nested type"""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_dc)")
        # print("f_dc:\n", stdout)
        assert stdout.startswith("def f_dc(a: Foo) -> Bar:")
        assert '    """Convert Foo to Bar."""' in stdout

        _out, stdout, _stderr = await sb.repl_command("show_definition(f_builtins)")
        # print("f_builtins:\n", stdout)
        assert stdout.startswith("def f_builtins(u: set[int], v: list[str]) -> dict[str, int]:")
        assert '    """Function using built-in generic types (set, list, dict)."""' in stdout

        # TODO: WE NEED TO FIX THIS TO REMOVE MODULE NAME FROM THE TYPE
        _out, stdout, _stderr = await sb.repl_command("show_definition(f_container_dc)")
        # print("f_container_dc:\n", stdout)
        assert stdout.startswith("def f_container_dc(items: set[Foo]) -> list[Bar]:")
        assert '    """Function with dataclass generics."""' in stdout


def test_method_stub_generation():
    from sandbox.guest import stubs

    assert stubs is not None

    class X:
        def foo(self) -> None: ...

        def bar(self, x: int) -> str: ...

        @classmethod
        def c_foo(cls) -> None: ...

        @staticmethod
        def s_foo() -> None: ...

    x = X()
    fmt_x_foo = stubs.format_definition(x.foo)
    assert fmt_x_foo == "def foo() -> None: ..."

    fmt_x_bar = stubs.format_definition(x.bar)
    assert fmt_x_bar == "def bar(x: int) -> str: ..."

    fmt_X_c_foo = stubs.format_definition(X.c_foo)
    fmt_x_c_foo = stubs.format_definition(x.c_foo)
    assert fmt_X_c_foo == "def c_foo() -> None: ..." == fmt_x_c_foo

    fmt_X_s_foo = stubs.format_definition(X.s_foo)
    fmt_x_s_foo = stubs.format_definition(x.s_foo)
    assert fmt_X_s_foo == "def s_foo() -> None: ..." == fmt_x_s_foo


def test_overload_stub_generation():
    from typing import overload

    from sandbox.guest import stubs

    assert stubs is not None

    @overload
    def foo(x: int) -> int:
        """i love ints"""
        ...

    @overload
    def foo(x: str) -> str:
        """i love strings"""
        ...

    def foo(x: int | str) -> int | str:
        """i love ints and strings"""
        return x

    fmt_fn_overloaded = stubs.format_definition(foo)
    assert (
        fmt_fn_overloaded
        == dedent('''
        @overload
        def foo(x: int) -> int:
            """i love ints"""

        @overload
        def foo(x: str) -> str:
            """i love strings"""
        ''').strip()
    )

    class X:
        @overload
        def foo(self, x: int) -> int:
            """i love ints"""
            ...

        @overload
        def foo(self, x: str) -> str:
            """i love strings"""
            ...

        def foo(self, x: int | str) -> int | str:
            """i love ints and strings"""
            return x

    x = X()
    fmt_x_foo = stubs.format_definition(x.foo)
    assert (
        fmt_x_foo
        == dedent('''
        @overload
        def foo(x: int) -> int:
            """i love ints"""

        @overload
        def foo(x: str) -> str:
            """i love strings"""
        ''').strip()
    )


@pytest.mark.asyncio
async def test_overload_stub_generation_warp():
    from typing import overload

    asyncio.get_running_loop().set_debug(False)

    # START: globals
    @overload
    def foo(x: int) -> int:
        """i love ints"""
        ...

    @overload
    def foo(x: str) -> str:
        """i love strings"""
        ...

    def foo(x: int | str) -> int | str:
        """i love ints and strings"""
        return x

    class X:
        @overload
        def bar(self, x: int) -> int:
            """i love ints"""
            ...

        @overload
        def bar(self, x: str) -> str:
            """i love strings"""
            ...

        def bar(self, x: int | str) -> int | str:
            """i love ints and strings"""
            return x

    x = X()
    # END: globals

    async with SandboxTestContext('test_overload_stub_generation_warp', foo, X=X, x=x) as sb:
        _, fmt_fn_overloaded, _ = await sb.repl_command("show_definition(foo)")

        assert fmt_fn_overloaded == (
            dedent('''
            @overload
            def foo(x: int) -> int:
                """i love ints"""

            @overload
            def foo(x: str) -> str:
                """i love strings"""
            ''').strip()
        )

        _, fmt_x_bar, _ = await sb.repl_command("show_definition(x.bar)")
        assert (
            fmt_x_bar
            == dedent('''
            @overload
            def bar(x: int) -> int:
                """i love ints"""

            @overload
            def bar(x: str) -> str:
                """i love strings"""
            ''').strip()
        )

        _, fmt_X, _ = await sb.repl_command("show_definition(X)")
        assert fmt_X == (
            dedent('''
            class X:
                def __init__(self): ...
                @overload
                def bar(self, x: int) -> int:
                    """i love ints"""

                @overload
                def bar(self, x: str) -> str:
                    """i love strings"""
            ''').strip()
        )


if __name__ == '__main__':
    pytest.main(['-vv', '-s', __file__])
