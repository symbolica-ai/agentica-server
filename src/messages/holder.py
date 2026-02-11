from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from typing import Any, Callable

type FilterFn[M] = Callable[[M], bool] | None


class Holder[K, M]:
    to_json: Callable[[M], dict[str, Any]]
    _messages: dict[K, deque[M]]
    _listeners: dict[K, deque[Callable[[M], None]]]
    _global_listeners: deque[Callable[[M], None]]

    def __init__(self, to_json: Callable[[M], dict[str, Any]]):
        self.to_json = to_json
        self._messages = defaultdict(lambda: deque())
        self._listeners = defaultdict(lambda: deque())
        self._global_listeners = deque()

    def add_listener(self, key: K, listener: Callable[[M], None]) -> None:
        self._listeners[key].append(listener)

    def add_global_listener(self, listener: Callable[[M], None]) -> None:
        self._global_listeners.append(listener)

    def add(self, key: K, message: M) -> None:
        for listener in self._listeners[key]:
            listener(message)
        for listener in self._global_listeners:
            listener(message)
        self._messages[key].append(message)

    def get_by_key(self, key: K, filter_fn: FilterFn[M] = None) -> Iterator[M]:
        messages = (x for x in self._messages[key])
        if filter_fn is not None:
            return (m for m in messages if filter_fn(m))
        return messages

    def get_all(self, filter_fn: FilterFn[M] = None) -> Iterator[M]:
        messages = (message for messages in self._messages.values() for message in messages)
        if filter_fn is not None:
            return (m for m in messages if filter_fn(m))
        return messages

    def get_all_keys(self) -> Iterable[K]:
        return self._messages.keys()

    def get_json_by_key(self, key: K, filter_fn: FilterFn[M] = None) -> Iterator[dict[str, Any]]:
        return (self.to_json(message) for message in self.get_by_key(key, filter_fn))

    def get_json_all(self, filter_fn: FilterFn[M] = None) -> Iterator[dict[str, Any]]:
        return (self.to_json(message) for message in self.get_all(filter_fn))

    def remove_key(self, key: K) -> None:
        """Remove all messages and listeners for a specific key."""
        _ = self._messages.pop(key, None)
        _ = self._listeners.pop(key, None)
