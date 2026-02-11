"""
Vibe coded stub printer for showing python values to LLMs.
"""

import inspect
import sys
import types
import typing
from collections.abc import Collection, Mapping
from typing import Protocol

import typing_extensions

_SENTINEL = object()
_DEFAULT_PREVIEW_LEN = 120
tab = "    "


def _safe_repr(obj: object, max_len: int = _DEFAULT_PREVIEW_LEN) -> str:
    """repr(obj) but truncated and made single-line."""
    try:
        r = repr(obj)
    except Exception as exc:
        r = f"<unrepresentable {type(obj).__name__}: {exc}>"
    r = r.replace("\n", " ").strip()
    return (r[: max_len - 1] + "..." + r[-1]) if len(r) > max_len else r


def _format_docstring(obj: object, indent: str = "") -> str:
    """Extract and format docstring with proper indentation."""
    try:
        docstring = inspect.getdoc(obj)
        if not docstring:
            return ""

        # Split into lines and add proper indentation
        lines = docstring.split('\n')
        # First line goes on the same line as the triple quotes
        if len(lines) == 1:
            return f'{indent}"""{lines[0]}"""'

        # Multi-line docstring
        formatted_lines = [f'{indent}"""{lines[0]}']
        for line in lines[1:]:
            # Add indent to each subsequent line
            formatted_lines.append(f'{indent}{line}' if line.strip() else f'{indent}')
        formatted_lines.append(f'{indent}"""')
        return '\n'.join(formatted_lines)
    except Exception:
        return ""


_BUILTIN_TYPES = frozenset(
    {'int', 'str', 'float', 'bool', 'dict', 'list', 'tuple', 'set', 'bytes', 'NoneType'}
)


def clean_type_name(type_obj: object, context: dict[str, typing.Any] | None = None) -> str:
    """Get a clean, readable type name from a type object and add it to context."""
    if isinstance(type_obj, typing.ForwardRef):
        return type_obj.__forward_arg__
    if isinstance(type_obj, str):
        return repr(type_obj)
    if isinstance(type_obj, typing.TypeAliasType):
        return type_obj.__name__

    if context is None:
        context = dict()

    if type_obj is None or type_obj is types.NoneType:
        return "None"

    # Handle union types
    try:
        origin = typing.get_origin(type_obj)
    except Exception:
        origin = None
    if origin is typing.Union or origin is types.UnionType:
        try:
            args = typing.get_args(type_obj)
        except Exception:
            args = getattr(type_obj, '__args__', ())
        parts = [clean_type_name(arg, context) for arg in args]
        return " | ".join(parts)

    # Handle generic types like List[str], Dict[str, int]
    origin = getattr(type_obj, '__origin__', None)
    if origin is not None:
        # For parameterized generics, we want the base type (List, not List[str])
        name = origin.__name__
        if hasattr(type_obj, '_name') and getattr(type_obj, '_name'):
            name = getattr(type_obj, '_name')
            # Try to get the actual typing class
            try:
                import typing as typing_mod

                actualtypingype = getattr(typing_mod, name, None)
                if actualtypingype:
                    context[name] = actualtypingype
            except Exception:
                context[name] = type_obj

        # Also process any type arguments
        args = getattr(type_obj, '__args__', ())
        args = [clean_type_name(arg, context) for arg in args]

        # Return the clean representation
        return f"{name}[{', '.join(args)}]".replace("typing.", "")

    # Handle built-in types
    if hasattr(type_obj, '__name__'):
        name = getattr(type_obj, '__name__')
        if name in _BUILTIN_TYPES:
            return name
        else:
            # Add non-builtin types to context
            context[name] = type_obj
            return name

    # Handle typing constructs like Optional (when not parameterized)
    if hasattr(type_obj, '_name'):
        name = getattr(type_obj, '_name', None)
        if name and name not in _BUILTIN_TYPES:
            try:
                import typing as typing_mod

                actualtypingype = getattr(typing_mod, name, None)
                if actualtypingype:
                    context[name] = actualtypingype
                else:
                    context[name] = type_obj
            except Exception:
                context[name] = type_obj
            return name

    # Fallback: use repr and clean it up
    type_repr = repr(type_obj)
    type_repr = type_repr.replace('__main__.', '')
    type_repr = type_repr.replace("typing.", "")
    type_repr = type_repr.replace("<class '", "").replace("'>", "")

    return "None" if type_repr == "NoneType" else type_repr


def _format_collection_sample(items: list, max_items: int = 3) -> str:
    """Format a sample of collection items with ellipsis if truncated."""
    sample_items = [_safe_repr(x) for x in items[:max_items]]
    sample = ", ".join(sample_items)
    return f"{sample}, ..." if len(items) > max_items else sample


def _format_dict_stub(name: str, val: Mapping) -> str:
    """Format a dictionary stub with proper syntax."""
    if len(val) == 0:
        return f"{name}: {type(val).__name__} = {{}}"

    items = list(val.items())
    samples = [f"{_safe_repr(k)}: {_safe_repr(v)}" for k, v in items[:3]]
    sample_str = ", ".join(samples)
    if len(val) > 3:
        sample_str += ", ..."
    return f"{name}: {type(val).__name__} = {{{sample_str}}}"


def _format_sequence_stub(name: str, val: Collection) -> str:
    """Format list/tuple stub with proper brackets."""
    istypinguple = isinstance(val, tuple)
    if len(val) == 0:
        brackets = "()" if istypinguple else "[]"
        return f"{name}: {type(val).__name__} = {brackets}"

    sample = _format_collection_sample(list(val))
    brackets = f"({sample})" if istypinguple else f"[{sample}]"
    return f"{name}: {type(val).__name__} = {brackets}"


def _format_set_stub(name: str, val: set) -> str:
    """Format set stub with proper syntax."""
    if len(val) == 0:
        return f"{name}: {type(val).__name__} = set()"

    sample = _format_collection_sample(list(val))
    return f"{name}: {type(val).__name__} = {{{sample}}}"


def _format_function_stub(
    name: str,
    func: object,
    context: dict[str, typing.Any],
    indent: str = "",
    bound_instance: object | None = None,
) -> str:
    """Format a function or method stub with type annotations and docstring."""
    try:
        is_method = False
        is_class_method = False
        # If we're formatting a bound method (instance/class), remember the bound object
        if bound_instance is None and getattr(func, "__self__", None) is not None:
            bound_instance = getattr(func, "__self__", None)
            is_class_method = isinstance(bound_instance, type)
            is_method = not is_class_method

        if (overloads := typing.get_overloads(func)) and func not in overloads:
            return "\n\n".join(
                f"{indent}@overload\n{_format_function_stub(name, overload, context, indent, bound_instance=bound_instance)}"
                for overload in overloads
            )

        ann = get_type_hints(func)
        sig = inspect.signature(func)  # type: ignore[arg-type]

        # Check if the function is async
        is_async = inspect.iscoroutinefunction(func)
        async_prefix = "async " if is_async else ""

        # Re-insert annotations that were stripped off the signature by typing
        params = []
        # Only omit the first parameter when formatting unbound overload functions
        # in the context of an originally bound method. For already-bound methods,
        # inspect.signature() has already removed the implicit first param.
        omit_first_param = bound_instance is not None and not is_method

        is_init = name == '__init__'
        for idx, p in enumerate(sig.parameters.values()):
            # If the original callable was bound, mimic bound-signature by dropping first positional param
            if (
                omit_first_param
                and idx == 0
                and (
                    p.kind == inspect.Parameter.POSITIONAL_ONLY
                    or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
            ):
                continue
            ann_str = ""

            p_name = p.name
            if p_name.startswith("___"):
                # warp internal
                continue

            if p_name in ann:
                type_obj = ann[p_name]
                type_repr = clean_type_name(type_obj, context)
                ann_str = f": {type_repr}"

            default = ""
            if p.default is not inspect._empty:
                default = f" = {_safe_repr(p.default)}"

            stars = ""
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                stars = "*"
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                stars = "**"

            params.append(f"{stars}{p_name}{ann_str}{default}")

        ret = ""
        if 'return' in ann:
            ret_obj = ann['return']
            ret_repr = clean_type_name(ret_obj, context)
            ret = f" -> {ret_repr}"

        param_s = ", ".join(params)
        docstring = _format_docstring(func, indent + "    ")
        signature = f"{indent}{async_prefix}def {name}({param_s}){ret}:"

        return f"{signature}\n{docstring}" if docstring else f"{signature} ..."
    except Exception:
        return f"{indent}def {name}(...): ..."


def _stub_for_value(
    name: str,
    val: object,
    context: dict[str, typing.Any],
    where_context: dict[str, typing.Any] | None = None,
) -> str:
    """Dispatch to the right stub builder based on *val*'s type.

    Args:
        name: Variable name
        val: Value to create stub for
        context: Context for all referenced types (for type annotations)
        where_context: Optional separate context for types that should go in "where" clause (object instances only)
    """
    # Add the value itself to context if it's a class, function, or type
    if isinstance(val, type) or inspect.isfunction(val) or inspect.ismethod(val):
        context[name] = val

    if inspect.isfunction(val) or inspect.ismethod(val):
        return _format_function_stub(name, val, context)
    elif isinstance(val, type):
        ignored_bases: set[type] = set()
        added_bases: set[str] = set()
        if typing_extensions.is_typeddict(val):
            ignored_bases.add(dict)
            added_bases.add('TypedDict')
        bases = [b.__name__ for b in val.__bases__ if b is not object and b not in ignored_bases]
        bases.extend(added_bases)
        base_s = f"({', '.join(bases)})" if bases else ""

        # Get the class docstring
        class_docstring = _format_docstring(val, "    ")

        # Check if this is a Protocol - Protocols shouldn't have __init__ in stubs
        try:
            is_protocol = isinstance(val, type) and issubclass(val, Protocol)
        except:
            is_protocol = False

        # Get methods and attributes from the class
        methods = []

        seen: set[str] = set()
        annos = []

        get_members = dict(inspect.getmembers(val))
        members = {k: get_members[k] for k in val.__annotations__ if k in get_members}
        members |= {k: v for k, v in get_members.items() if k not in members}

        for attr_name, attr_val in members.items():
            allow_list = {'__getitem__', '__setitem__', '__delitem__', '__contains__'}
            if not is_protocol:
                allow_list.update({'__init__', '__call__'})
            if attr_name.startswith("_") and attr_name not in allow_list:
                continue  # Skip private/special methods except __init__

            try:
                if inspect.isfunction(attr_val) or inspect.ismethod(attr_val):
                    # Use the shared function stub formatter with indentation
                    method_stub = _format_function_stub(attr_name, attr_val, context, "    ")
                    methods.append(method_stub)
                elif not callable(attr_val):
                    # Class attribute
                    seen.add(attr_name)
                    type_name = clean_type_name(val.__annotations__.get(attr_name, type(attr_val)))
                    repr_val = _safe_repr(attr_val)
                    member_str = f"{tab}{attr_name}: {type_name}"
                    if not (
                        # type(attr_val) is types.MemberDescriptorType or
                        # type(attr_val) is types.GetSetDescriptorType or
                        repr_val.startswith('<attribute')
                    ):
                        member_str += f" = {repr_val}"
                    annos.append(member_str)
            except Exception:
                # Fallback for problematic attributes
                type_name = getattr(type(attr_val), '__name__', 'object')
                methods.append(f"{tab}{attr_name}: {type_name}")

        # Collect annotations from this class and all bases (for Protocol inheritance)
        # This ensures inherited Protocol fields are shown in stubs
        all_annotations = {}
        for base in reversed(
            val.__mro__[:-1]
        ):  # Reverse MRO to get proper override order, skip 'object'
            if hasattr(base, '__annotations__'):
                all_annotations.update(base.__annotations__)

        for anno_name, anno_val in all_annotations.items():
            if anno_name in seen:
                continue
            seen.add(anno_name)
            anno_line = f"{tab}{anno_name}: {clean_type_name(anno_val)}"
            if anno_name in val.__dict__:
                anno_line += f" = {_safe_repr(getattr(val, anno_name))}"
            annos.append(anno_line)

        # Build the class body in the correct order:
        # 1. Docstring (if present)
        # 2. Field annotations
        # 3. Methods
        body_parts = []

        if class_docstring:
            body_parts.append(class_docstring)

        if annos:
            body_parts.append("\n".join(annos))

        if methods:
            body_parts.append("\n".join(methods))

        if not body_parts:
            body_parts.append(f"{tab}...")

        # Add appropriate gaps between sections
        body_str = "\n"
        for i, part in enumerate(body_parts):
            if i > 0:
                # Add extra newline between sections (docstring->fields, fields->methods)
                body_str += "\n"
            body_str += part

        return f"class {name}{base_s}:{body_str}"
    elif isinstance(val, (int, float, str, bool, bytes, complex)):
        return f"{name}: {type(val).__name__} = {_safe_repr(val)}"
    elif isinstance(val, Mapping) and not isinstance(val, (str, bytes)):
        return _format_dict_stub(name, val)
    elif isinstance(val, Collection) and not isinstance(val, (str, bytes)):
        if isinstance(val, (list, tuple)):
            return _format_sequence_stub(name, val)
        elif isinstance(val, set):
            return _format_set_stub(name, val)
        else:
            # Other collections - fallback to generic format
            sample = _format_collection_sample(list(val))
            extra = f"[{sample}]" if sample else ""
            return f"{name}: {type(val).__name__} (len={len(val)}) {extra}"
    else:
        # For object instances, add the class to where_context (if provided) or context
        val_type = type(val)
        type_name = val_type.__name__

        # Special case for modules - keep the descriptive format
        if isinstance(val, types.ModuleType):  # Check if it's a module
            return f"{name}: {type_name} = {_safe_repr(val)}"

        # Add the class to the appropriate context
        if type_name not in _BUILTIN_TYPES:
            # If where_context is provided, this is for "where" clause tracking
            if where_context is not None:
                where_context[type_name] = val_type
            # Always add to regular context too (for type annotations)
            context[type_name] = val_type

        return f"{name}: {type_name} = {_safe_repr(val)}"


def emit_stubs(
    ns: dict[str, typing.Any] | None = None,
    *,
    max_lines: int | None = None,
    exclude_private: bool = True,
    exclude_names: set[str] | None = None,
    sort_items: bool = True,
) -> tuple[str, dict[str, typing.Any]]:
    """
    Return stubs and required context for every binding in *ns* (defaults to caller's globals()).

    Parameters
    ----------
    ns : dict | None
        Namespace to inspect.  If None, inspect the caller's frame.
    max_lines : int | None
        Truncate the output to at most *max_lines* lines (helps huge notebooks).
    exclude_private : bool
        Skip names that start with an underscore.
    exclude_names : set[str] | None
        Exclude explicit names.
    sort_items : bool
        Whether to sort items by key

    Returns
    -------
    tuple[str, dict[str, Any]]
        A tuple of (stub_string, required_context) where required_context
        contains all types and objects referenced in the stubs.
    """
    if ns is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            ns = frame.f_back.f_globals
        else:
            ns = {}

    required_context: dict[str, typing.Any] = dict()
    where_context: dict[str, typing.Any] = dict()
    lines: list[str] = []
    items = list(ns.items())
    if sort_items:
        items.sort(key=lambda item: item[0])
    for name, val in items:
        if exclude_private and name.startswith("_"):
            continue
        if exclude_names and name in exclude_names:
            continue
        lines.append(_stub_for_value(name, val, required_context, where_context))
        if max_lines is not None and len(lines) >= max_lines:
            break

    # Generate class definitions for types from object instances only (where_context)
    class_definitions: list[str] = []
    seen_classes: set[str] = set()

    items = list(where_context.items())
    if sort_items:
        items.sort(key=lambda item: item[0])
    for ctx_name, ctx_val in items:
        # Only generate definitions for classes that aren't already in the namespace
        if isinstance(ctx_val, type) and ctx_name not in ns and ctx_name not in seen_classes:
            seen_classes.add(ctx_name)
            # Create a temporary context for this class definition
            temp_context: dict[str, typing.Any] = {}
            class_stub = _stub_for_value(ctx_name, ctx_val, temp_context, None)
            class_definitions.append(class_stub)

    # Format the output
    result = "\n\n".join(lines)

    if class_definitions:
        result += "\n\nwhere\n\n" + "\n\n".join(class_definitions)

    return result, required_context


# --- Convenience one-liner ----------------------------------------------------
def print_stubs(
    ns: dict[str, typing.Any] | None = None,
    *,
    max_lines: int | None = None,
    exclude_private: bool = True,
):
    """Print stubs for the caller's globals()."""
    stubs, _ = emit_stubs(ns, max_lines=max_lines, exclude_private=exclude_private)
    print(stubs)


_context_so_far: dict[str, typing.Any] | None = None


def print_stubs_stateful(
    ns: dict[str, typing.Any] | None = None,
    *,
    max_lines: int | None = None,
    exclude_private: bool = True,
):
    global _context_so_far

    if ns is None:
        ns = _context_so_far

    stubs, context = emit_stubs(
        ns,
        max_lines=max_lines,
        exclude_private=exclude_private,
    )

    if _context_so_far is None:
        _context_so_far = context
    else:
        _context_so_far.update(context)

    print(stubs)


def show_definition(value: typing.Any) -> None:
    """
    Show the definition of a class, function, module or other object.
    """
    print(format_definition(value))


def format_definition(value: typing.Any) -> str:
    varname = '_'
    if hasattr(value, '__name__'):
        varname = value.__name__
    display, _ = emit_stubs({varname: value}, exclude_private=False)
    return display


_previous_locals: dict[str, typing.Any] | None = None
_hidden_locals: set[str] = set()


def before_locals_population():
    global _previous_locals
    _previous_locals = sys._getframe(1).f_globals.copy()


def locals_delta():
    global _previous_locals
    locs = sys._getframe(1).f_locals.copy()
    if _previous_locals is None:
        _previous_locals = dict()
    return {
        local: val
        for local, val in locs.items()
        if (local not in _previous_locals or _previous_locals[local] != val)
        and local not in _hidden_locals
    }


def hide_variable(name: str) -> None:
    _hidden_locals.add(name)


def get_type_hints(func: types.FunctionType) -> dict[str, typing.Any]:
    try:
        return typing.get_type_hints(func, include_extras=True)
    except NameError:
        if hasattr(func, '__annotations__'):
            return func.__annotations__
        return {}


def print_annotations(func: types.FunctionType) -> None:
    annos = get_type_hints(func).copy()
    annos.pop('return', None)

    params = []
    for p in inspect.signature(func).parameters.values():
        if p.name in annos:
            type_obj = annos[p.name]
            type_repr = clean_type_name(type_obj)
        else:
            type_repr = "Any"

        anno_str = f": {type_repr}"

        params.append(f"{p.name}{anno_str}")

    print('(' + ', '.join(params) + ')')
