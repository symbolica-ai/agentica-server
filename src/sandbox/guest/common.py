import sys
from typing import Any

__all__ = [
    "_is_instance",
    "_warp_aware_vars",
    "_generic_repr",
    "_try_repr",
    "_repr_trunc",
]


def _is_instance(obj: object, typ: type) -> bool:
    import typeguard

    try:
        typeguard.check_type(obj, typ)
    except typeguard.TypeCheckError:
        return False
    except:
        pass
    return True


def _warp_aware_vars(obj: object) -> dict[str, object]:
    v: dict[str, Any] = {}
    try:
        v = vars(obj).copy()
    except:
        try:
            v = {'___grid___': None, '___keys___': obj.___keys___}
        except:
            pass
    try:
        _ = v.pop('___grid___', None)
        keys = v.pop('___keys___', set())
        v |= {k: getattr(obj, k) for k in keys if hasattr(obj, k)}
    except:
        pass
    return v


def _generic_repr(obj: object) -> str:
    return (
        f"{obj.__class__.__qualname__}("
        + ", ".join(f"{k!s}={v!r}" for k, v in _warp_aware_vars(obj).items())
        + ")"
    )


def _try_repr(obj: object) -> str:
    try:
        return repr(obj)
    except TypeError:
        return _generic_repr(obj)


_try_repr.__name__ = repr.__name__
_try_repr.__qualname__ = repr.__qualname__
_try_repr.__doc__ = repr.__doc__
_try_repr.__module__ = repr.__module__


MAX_EXECUTION_TEXT_LENGTH = 80
MAX_EXECUTION_TEXT_WIDTH = 1000


def _repr_trunc(value: object, depth: int = 0) -> str:
    above_globals = sys._getframe(depth + 1).f_globals
    max_execution_text_length = above_globals.get(
        '_max_execution_text_length', MAX_EXECUTION_TEXT_LENGTH
    )
    max_execution_text_width = above_globals.get(
        '_max_execution_text_width', MAX_EXECUTION_TEXT_WIDTH
    )

    text = _try_repr(value)
    texts = text.split("\n")
    flag = False
    for i, t in enumerate(texts):
        if len(t) > max_execution_text_width:
            t = (
                t[:max_execution_text_width]
                + f"... [truncated, {len(t) - max_execution_text_width} characters]"
            )
        texts[i] = t
        if i >= max_execution_text_length:
            flag = True
    text = "\n".join(texts)
    if flag:
        text += f"\n... [truncated, {len(texts) - MAX_EXECUTION_TEXT_LENGTH} lines omitted]"
    return text
