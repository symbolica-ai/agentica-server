import asyncio
import traceback

from agentica_internal.core.log import should_log_tag
from agentica_internal.core.print import tprint
from agentica_internal.core.utils import is_type_like
from json_schema import (
    SchemaInfoCallable,
    SchemaInfoObject,
    SchemaResult,
    SchemaResult_ExecutionError,
    SchemaResult_GenerationError,
    SchemaResult_NotFound,
    SchemaResult_Ok,
    SchemaResult_ValidationError,
    execute_callable,
    get_schema_info_callable,
    get_schema_info_object,
    schema_result_to_str,
)
from pydantic import ValidationError

from .agent_error import AgentError

__all__ = [
    'clear_schema_info_cache',
    "repl_get_schema_info_object",
    "repl_get_schema_info_callable",
    "repl_execute_callable",
    "schema_result_to_str",
]

_cached_sic: dict[tuple[int, str], SchemaInfoCallable] = dict()
_cached_sio: dict[tuple[int, str], SchemaInfoObject] = dict()


def clear_schema_info_cache():
    _cached_sic.clear()
    _cached_sio.clear()


def repl_get_schema_info_object(
    repl_uuid: int, repl_local_scope: dict[str, object], obj_name: str
) -> SchemaResult:
    global _cached_sio
    cache_key = repl_uuid, obj_name
    if cache_key in _cached_sio:
        return SchemaResult_Ok(*_cached_sio[cache_key].to_wit_data())
    if obj_name not in repl_local_scope:
        return SchemaResult_NotFound(f"NotFound({obj_name})")
    obj = repl_local_scope[obj_name]
    if not is_type_like(obj):
        return SchemaResult_GenerationError(f"InvalidObject({obj_name}, {type(obj)})")
    try:
        sio = get_schema_info_object(obj)
    except BaseException as e:
        return SchemaResult_GenerationError(str(e) + "\n" + traceback.format_exc())
    _cached_sio[cache_key] = sio
    return SchemaResult_Ok(*sio.to_wit_data())


def repl_get_schema_info_callable(
    repl_uuid: int, repl_local_scope: dict[str, object], callable_name: str
) -> SchemaResult:
    global _cached_sic
    cache_key = repl_uuid, callable_name
    if cache_key in _cached_sic:
        return SchemaResult_Ok(*_cached_sic[cache_key].to_wit_data())
    obj = repl_local_scope.get(callable_name)
    if obj is None or not callable(obj):
        return SchemaResult_GenerationError(f"({callable_name}, {type(obj)})")
    try:
        sic = get_schema_info_callable(obj)
    except BaseException as e:
        return SchemaResult_GenerationError(str(e) + "\n" + traceback.format_exc())
    _cached_sic[cache_key] = sic
    return SchemaResult_Ok(*sic.to_wit_data())


def repl_execute_callable(
    repl_uuid: int,
    repl_local_scope: dict[str, object],
    obj_name: str,
    content: str,
    assign_to: str,
) -> SchemaResult:
    log = should_log_tag(False, {'json', 'repl_execute_callable'})
    cache_key = repl_uuid, obj_name
    tprint(f"repl_execute_callable: getting schema info for {obj_name!r}") if log else None
    global _cached_sic
    # Stub: return a default outcome (use enum for local runner; componentizer will map)
    if cache_key not in _cached_sic:
        return SchemaResult_ExecutionError(f"callable {obj_name} not found")
    sic = _cached_sic[cache_key]
    tprint(f"repl_execute_callable: schema info = {sic}") if log else None
    try:
        tprint(f"repl_execute_callable: asyncio.run") if log else None
        result = asyncio.run(
            execute_callable(
                content=content,
                allow_partial=False,
                sic=sic,
            )
        )
        tprint(
            f"repl_execute_callable: got result of type {type(result).__name__!r}"
        ) if log else None
        repl_local_scope[assign_to] = result
        return SchemaResult_Ok("execution-result", str(result))
    except ValidationError as e:
        tprint(f"repl_execute_callable: caught ValidationError") if log else None
        return SchemaResult_ValidationError(str(e) + "\n" + traceback.format_exc())
    except AgentError as e:
        tprint(f"repl_execute_callable: caught AgentError") if log else None
        raise e
    except BaseException as e:
        tprint(f"repl_execute_callable: caught exception {type(e).__name__}") if log else None
        return SchemaResult_ExecutionError(str(e) + "\n" + traceback.format_exc())
