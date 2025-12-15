from functools import wraps
from typing import Callable, Generator, get_origin

__all__ = ['do']


def do[M, **I, O: Generator](monad: type[M]) -> Callable[[Callable[I, O]], Callable[I, M]]:
    """
    Turn a generator into do-notation for a monad.

    ```python
    m = a >> (lambda x: b >> (lambda y: c >> (lambda z: pure((x, y, z)))))
    ```
    is equivalent to
    ```python
    @do(MonadT)
    def m():
        x = yield a
        y = yield b
        z = yield c
        return (x, y, z)
    ```
    """
    if t := get_origin(monad):
        monad = t  # the monad class

    pure = getattr(monad, 'Pure')
    immediate = getattr(monad, 'immediate')

    def _decorator(func: Callable[I, O]) -> Callable[I, M]:
        @wraps(func)
        def wrapped(*args, **kwargs) -> M:
            def build_monad():
                gen = func(*args, **kwargs)  # is a generator

                # `m` is the last yielded monad.
                # the next `.send()` will send back the unwrapped value of `m`.

                def send(x):
                    try:
                        m = gen.send(x)
                        return m.bind(send)
                    except StopIteration as e:
                        if e.value is None:
                            # no return, just produce the last sent value
                            return pure(x)
                        elif isinstance(e.value, monad):
                            # the returned value is a monad
                            return e.value
                        else:
                            # we wrap the returned value into the monad
                            return pure(e.value)

                return send(None)

            return immediate(build_monad)

        return wrapped

    return _decorator
