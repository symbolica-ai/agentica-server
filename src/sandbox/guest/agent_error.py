__all__ = [
    'AgentError',
]


class AgentError(BaseException):
    """An Exception class which is used to wrap an exception which will later be raised."""

    inner_exception: BaseException

    def __init__(self, inner_exception: BaseException | str):
        if not isinstance(inner_exception, BaseException):
            # In case the agent does not wrap the exception properly.
            inner_exception = RuntimeError(inner_exception)
        self.inner_exception = inner_exception
        super().__init__()

    def __repr__(self) -> str:
        return "AgentError<" + repr(self.inner_exception) + ">"
