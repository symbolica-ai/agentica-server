from collections.abc import Callable
from sys import getrecursionlimit, setrecursionlimit
from types import EllipsisType, NoneType, NotImplementedType
from typing import Any, Sequence

from agentica_internal.core.anno import ANNOS, anno_str
from agentica_internal.warpc.predicates import is_strtup
from agentica_internal.warpc.resource.handle import is_virtual_object

__all__ = ['safe_repr']


type ToStr = Callable[[Any, int], str]


DEFAULT_LIMIT = 256


def safe_repr(root: Any, allow_rpc: bool, limit: int) -> str:
    atom_types = str, bytes, type, bool, int, float, NoneType, NotImplementedType, EllipsisType

    if type(root) in atom_types:
        return atom_repr(root, limit)

    if allow_rpc and is_virtual_object(root):
        result = ''
        try:
            result = repr(root)
        except Exception:
            pass
        if type(result) is str and result:
            result = result.replace("\n", " ").strip()
            if len(result) > limit:
                result = result[: limit - 1] + ".." + result[-1]
            return result

    guard: set[int] = set()

    def fmt(obj: Any, lim: int) -> str:
        cls = type(obj)
        if cls in (bool, int, float, NoneType, NotImplementedType, EllipsisType):
            return repr(obj)
        if cls in atom_types or isinstance(obj, type):
            return atom_repr(obj, lim)
        if cls in (list, tuple, set, frozenset, dict):
            i = id(obj)
            if i in guard:
                return '..'
            guard.add(i)
            res = dict_repr(obj, fmt, lim) if cls is dict else sequence_repr(obj, fmt, lim)
            guard.discard(i)
            return res
        if cls in ANNOS:
            return anno_str(obj)
        if lim >= 8 and not is_virtual_object(obj):
            attrs = None
            if isinstance(fields := getattr(cls, '__dataclass_fields__', None), dict):
                attrs = tuple(k for k, f in fields.items() if getattr(f, 'repr', False))
            if slots := getattr(cls, '__slots__', None):
                if is_strtup(slots):
                    attrs = slots
            if attrs:
                i = id(obj)
                if i in guard:
                    return '..'
                guard.add(i)
                res = attrs_repr(obj, fmt, attrs, limit)
                guard.discard(i)
                return res
        return f'<{cls.__name__!r} object>'

    prev_rec_limit = getrecursionlimit()
    setrecursionlimit(SYS_REC_LIMIT)
    result = fmt(root, limit)
    setrecursionlimit(prev_rec_limit)

    return result


def attrs_repr(obj: Any, fmt: ToStr, attrs: Sequence[str], limit: int) -> str:
    strs = []
    add = strs.append
    cls_name = type(obj).__name__
    limit -= len(cls_name) + 2
    if limit < MAX_DICT_EXTRA:
        return f'{cls_name}(..)'
    for k in attrs:
        try:
            v = obj_get(obj, k)
        except:
            continue
        if limit > 0:
            f_val = fmt(v, min(limit, MAX_VAL_LIMIT))
            f_item = f'{k}={f_val}'
            limit -= len(f_item) + 2
            add(f_item)
        elif MAX_DICT_EXTRA <= limit <= 2:
            f_val = repr(v) if type(v) is int and -9 <= v <= 9 else '..'
            f_item = f'{k}={f_val}'
            limit -= len(f_item) + 2
            add(f_item)
        else:
            break
    if len(strs) == len(attrs):
        return f'{cls_name}({commas(strs)})'
    else:
        return f'{cls_name}({commas(strs)}, ..)'


obj_get = object.__getattribute__


def sequence_repr(seq: Sequence, fn: ToStr, limit: int) -> str:
    k = (list, tuple, set, frozenset).index(type(seq))

    n = len(seq)
    if n == 0:
        return ('[]', 'tuple()', 'set()', 'frozenset()')[k]

    l = ('[', '(', '{', 'frozenset(')[k]
    r = (']', ')', '}', 'frozenset()')[k]

    if n > 1 and limit <= 2:
        name = ('list', 'tuple', 'set', 'frozenset')[k]
        return f'<{name} len={n}>' if limit else f'<{name}>'

    if n == 1:
        c = ',' if k == 1 else ''
        return f'{l}{fn(seq[0], limit)}{c}{r}'

    strs = []
    add = strs.append
    limit -= 2
    for elem in seq:
        if limit > 0:
            f_elem = fn(elem, min(limit, MAX_VAL_LIMIT))
            limit -= len(f_elem)
            add(f_elem)
        else:
            break

    if len(strs) == n:
        return f'{l}{commas(strs)}{r}'
    else:
        return f'{l}{commas(strs)}, ..{r} <{n} items>'


MAX_DICT_EXTRA = -10
MAX_KEY_LIMIT = 32
MAX_VAL_LIMIT = 64


def dict_repr(d: dict, fmt: ToStr, limit: int) -> str:
    n = len(d)
    if n == 0:
        return '{}'

    if n > 1 and limit <= 2:
        return f'<dict len={n}>'

    if n == 1:
        k, v = list(d.items())[0]
        f_key = fmt(k, limit)
        limit -= len(f_key)
        f_val = fmt(v, limit)
        return f'{{{f_key}: {f_val}}}'

    strs = []
    add = strs.append
    limit -= 2

    is_rec = n < 16 and all(type(k) is str and k.isascii() for k in d)
    if is_rec:
        for k, v in d.items():
            if limit > 0 and is_rec:
                f_val = fmt(v, min(limit, MAX_VAL_LIMIT))
                f_item = f'{k}={f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            elif limit > 0:
                f_key = fmt(k, min(limit, MAX_KEY_LIMIT))
                f_val = fmt(v, min(limit, MAX_VAL_LIMIT))
                f_item = f'{f_key}: {f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            elif MAX_DICT_EXTRA <= limit <= 2 and type(k) is str:
                f_val = repr(v) if type(v) is int and -9 <= v <= 9 else '..'
                f_item = f'{k}={f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            else:
                break
        if len(strs) == n:
            return f'dict({commas(strs)})'
        else:
            return f'dict({commas(strs)}, ..) <{n} items>'
    else:
        for k, v in d.items():
            if limit > 0 and is_rec:
                f_val = fmt(v, min(limit, MAX_VAL_LIMIT))
                f_item = f'{k}={f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            elif limit > 0:
                f_key = fmt(k, min(limit, MAX_KEY_LIMIT))
                f_val = fmt(v, min(limit, MAX_VAL_LIMIT))
                f_item = f'{f_key}: {f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            elif MAX_DICT_EXTRA <= limit <= 2 and type(k) is str:
                f_key = fmt(k, min(limit, MAX_VAL_LIMIT))
                f_val = repr(v) if type(v) is int and -9 <= v <= 9 else '..'
                f_item = f'{f_key}: {f_val}'
                limit -= len(f_item) + 2
                add(f_item)
            else:
                break
        if len(strs) == n:
            return f'{{{commas(strs)}}}'
        else:
            return f'{{{commas(strs)}, ..}} <{n} items>'


def dc_repr(obj: Any, limit: int) -> str:
    cls = type(obj)
    if cls is type or isinstance(obj, type):
        return f'<class {cls.__name__!r}>'
    if cls in (bool, int, float, NoneType, NotImplementedType, EllipsisType):
        return repr(obj)
    if cls is str:
        n = len(obj)
        if n == 0:
            return "''"
        if n <= limit:
            return repr(obj)
        short = repr(obj[: limit + 15])
        if limit < 5:
            return f'<str>' if len(short) > 5 else short
        elif limit < 20:
            return f'<str len={n}>'
        elif len(obj) > 20:
            return f'{short[: limit - 10]}..{short[-10:]} <{n} chars>'
        elif len(obj) > 10:
            return f'{short[: limit - 5]}..{short[-5:]} <{n} chars>'
        else:
            return f'{short[: limit - 2]}..{short[-1]} <{n} chars>'
    if cls is bytes:
        n = len(obj)
        if n == 0:
            return "b''"
        if n <= limit:
            return repr(obj)
        if limit < 5:
            if len(obj) <= 7 and obj.isascii():
                return repr(obj)
            return f'<bytes>'
        return f'<bytes len={n}>'
    return f'<{cls.__name__!r} object>'


def atom_repr(obj: Any, limit: int) -> str:
    cls = type(obj)
    if cls is type or isinstance(obj, type):
        return f'<class {cls.__name__!r}>'
    if cls in (bool, int, float, NoneType, NotImplementedType, EllipsisType):
        return repr(obj)
    if cls is str:
        n = len(obj)
        if n == 0:
            return "''"
        if n <= limit:
            return repr(obj)
        short = repr(obj[: limit + 15])
        if limit < 5:
            return f'<str>' if len(short) > 5 else short
        elif limit < 20:
            return f'<str len={n}>'
        elif len(obj) > 20:
            return f'{short[: limit - 10]}..{short[-10:]} <{n} chars>'
        elif len(obj) > 10:
            return f'{short[: limit - 5]}..{short[-5:]} <{n} chars>'
        else:
            return f'{short[: limit - 2]}..{short[-1]} <{n} chars>'
    if cls is bytes:
        n = len(obj)
        if n == 0:
            return "b''"
        if n <= limit:
            return repr(obj)
        if limit < 5:
            if len(obj) <= 7 and obj.isascii():
                return repr(obj)
            return f'<bytes>'
        return f'<bytes len={n}>'
    return f'<{cls.__name__!r} object>'


commas = ', '.join

SYS_REC_LIMIT = getrecursionlimit()
