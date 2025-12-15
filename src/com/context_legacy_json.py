import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine, Literal, Optional

from json_schema import (
    SchemaInfoCallable,
    SchemaInfoObject,
    SchemaResult,
    SchemaResult_ExecutionError,
    SchemaResult_GenerationError,
    SchemaResult_NotFound,
    SchemaResult_Ok,
    SchemaResult_ValidationError,
)

from com.constraints import *
from com.context import *
from com.deltas import *
from com.model import *

__all__ = [
    'JSONExec',
    'Executable',
    'ObjectConstraint',
    'CallableConstraint',
    'EndGenObjectTypes',
    'EndGenCallableTypes',
]

# NOTE: this file is deprecated and will be removed in the future


if TYPE_CHECKING:
    from sandbox import Sandbox


@dataclass
class Executable:
    """
    An executable in the execution context.
    """

    name: str
    type = Literal['callable', 'object']
    id: str
    schema_info: SchemaInfoCallable | SchemaInfoObject | dict[str, Any] | None = None
    description: str | None = None

    def __init__(
        self,
        name: str,
        type: Literal['callable', 'object'],
        id: str | None = None,
        description: str | None = None,
    ):
        self.name = name
        self.type = type
        self.id = id if id is not None else str(uuid.uuid4())
        self.description = description

    async def schema(self, sandbox: 'Sandbox') -> dict[str, Any]:
        """
        Get the JSON schema and description of the executable from the sandbox.
        """
        if not await sandbox.repl_has_var(self.name, 'GLOBALS'):
            raise ValueError(f"Variable with name {self.name} is not in the sandbox's locals")

        var_info = await sandbox.repl_var_info(self.name)
        is_callable = var_info.is_callable
        is_inst_type = var_info.is_type_like

        if self.type == 'callable':
            if (not is_callable) or is_inst_type:
                print(f"{is_callable=}")
                print(f"{is_inst_type=}")
                raise ValueError(
                    f"Object with name {self.name} in the sandbox's locals is not a callable (or is a type)"
                )
            result = await sandbox.get_schema_info_callable(self.name)
        elif self.type == 'object':
            if not is_inst_type:
                raise ValueError(
                    f"Object with name {self.name} in the sandbox's locals is not a type"
                )
            result = await sandbox.get_schema_info_object(self.name)
        else:
            raise ValueError(f"Invalid type: {self.type}")

        match result:
            case SchemaResult_Ok(description, json_schema):
                if description:
                    self.description = description
                return json.loads(json_schema)
            case _:
                raise RuntimeError(
                    f"Schema generation failed for object {result.__class__.__name__} with name {self.name}: {result}"
                )

    async def exec(self, content: dict[str, Any] | str, sandbox: 'Sandbox', result_obj: str) -> str:
        """
        Execute the callable from the generated arguments in a sandbox.
        """
        # Get the schema of the executable from the sandbox to ensure the SIC is in the sandbox
        _ = await self.schema(sandbox)

        # Cannot execute object executables
        if self.type == 'object':
            raise ValueError(f"Cannot execute objects executables")

        # Convert the content to a string if it is not already
        if not isinstance(content, str):
            content = json.dumps(content)

        # Execute the callable in the sandbox
        result: SchemaResult = await sandbox.execute_callable(self.name, content, result_obj)

        match result:
            case SchemaResult_Ok(description, json_schema):
                return json_schema
            case SchemaResult_ValidationError(value):
                raise RuntimeError(
                    f"Validation failed for callable in sandbox with name {self.name}: {value}"
                )
            case SchemaResult_ExecutionError(value):
                raise RuntimeError(
                    f"Execution failed for callable in sandbox with name {self.name}: {value}"
                )
            case SchemaResult_NotFound(value):
                raise RuntimeError(f"Callable not found in sandbox with name {self.name}: {value}")
            case SchemaResult_GenerationError(value):
                raise RuntimeError(
                    f"Schema generation failed for callable in sandbox with name {self.name}: {value}"
                )
            case _:
                raise RuntimeError(
                    f"Execution failed for callable in sandbox with name {self.name}"
                )


class JSONExec:
    """
    Execution context.
    """

    executables: dict[str, Executable]
    code_buffer: str
    _system_exit_flag: bool
    _last_exec_exception: str | None

    _parent: Context | None
    _sandbox: 'Sandbox | None'
    _tasks: list[asyncio.Task[None]]

    def __init__(self, *, executables: dict[str, Executable], sandbox: 'Sandbox | None' = None):
        self.executables = executables
        self.code_buffer = ""
        self._system_exit_flag = False
        self._last_exec_exception = None
        self._parent = None
        self._tasks = []
        self._sandbox = sandbox

    @property
    def sandbox(self) -> 'Sandbox':
        return self._parent.sandbox if self._parent else self._sandbox

    async def update(
        self, *, globals_data: bytes = b'', locals_data: bytes = b'', json_mode: bool = False
    ) -> None:
        """
        Update the executables and sandbox with the new globals and locals.
        Note that the executables will not be initialised until their schema is fetched.

        Args:
            globals_data: The new globals as a bytes payload.
            locals_data: The new locals as a bytes payload.
            json_mode: Whether we are in JSON mode or not.
        """

        # Initialize the sandbox with the new globals and locals
        # _ = await self.repl_update(globals_data=globals_data, locals_data=locals_data)

        if not json_mode:
            return

        # Get the locals delta
        resources = self.sandbox.session_info.resources

        # Update the executables with the new locals if they are callable or object
        hidden = set()
        for name in resources.names:
            var_info = await self.sandbox.repl_var_info(name)
            if not var_info:
                continue
            if var_info.is_type_like:
                self.executables[name] = Executable(name, 'object')
                hidden.add(name)
            elif var_info.is_callable:
                self.executables[name] = Executable(name, 'callable')
                hidden.add(name)

        if hidden:
            _ = await self.sandbox.repl_hide_vars(hidden)

    @property
    def callable_executables(self) -> dict[str, Executable]:
        """
        Get all `CallableExecutable`s in the execution context.
        """
        return {e.name: e for e in self.executables.values() if e.type == 'callable'}

    @property
    def object_executables(self) -> dict[str, Executable]:
        """
        Get all `ObjectExecutable`s in the execution context.
        """
        return {e.name: e for e in self.executables.values() if e.type == 'object'}

    def _add_task(self, task: Coroutine[Any, Any, None]) -> None:
        self._tasks.append(asyncio.create_task(task, name="MonadTask"))

    def _log(self, name: str, *args: Any) -> None:
        if hasattr(self._parent, 'log'):
            try:
                self._add_task(self._parent.log(name, *args))
            except asyncio.CancelledError:
                pass


type GenType = Literal['json', 'custom']


@dataclass
class ObjectConstraint(Constraint):
    """
    A constraint for generating an instance of a type.
    """

    executables: list[Executable]
    type: GenType
    guided: bool
    name: str


@dataclass
class CallableConstraint(Constraint):
    """
    A constraint for generating arguments for a callable.
    """

    executable: Executable
    type: GenType
    guided: bool
    name: str
    allow_partial: bool = False


@dataclass
class EndGenObjectTypes(EndGen):
    """
    An object was generated.
    """

    constraint: ObjectConstraint
    content: dict[str, Any] | str


@dataclass
class EndGenCallableTypes(EndGen):
    """
    Callable arguments were generated.
    """

    constraints: list[CallableConstraint]
    ids: list[str]
    content: list[dict[str, Any] | str]
    results: Optional[list[Any]] = None
