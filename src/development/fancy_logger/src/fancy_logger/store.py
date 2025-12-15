from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


class LogStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._messages: list[dict[str, Any]] = []
        self._types: set[str] = set()
        # first-seen order tracking
        self._uid_first_index: dict[str, int] = {}
        self._uid_to_iid_first_index: dict[str, dict[str, int]] = defaultdict(dict)
        self._version: int = 0

    async def add(self, message: dict[str, Any]) -> None:
        # store the message and update indexes
        async with self._lock:
            idx = len(self._messages)
            self._messages.append(message)
            self._version += 1

            msg_type = message.get("type")
            if isinstance(msg_type, str):
                self._types.add(msg_type)

            uid = message.get("uid")
            if isinstance(uid, str):
                if uid not in self._uid_first_index:
                    self._uid_first_index[uid] = idx
                iid = message.get("iid")
                if isinstance(iid, str):
                    if iid not in self._uid_to_iid_first_index[uid]:
                        self._uid_to_iid_first_index[uid][iid] = idx

    async def types(self) -> list[str]:
        async with self._lock:
            return sorted(self._types)

    async def uids(self) -> list[str]:
        async with self._lock:
            return [uid for uid, _ in sorted(self._uid_first_index.items(), key=lambda kv: kv[1])]

    async def iids(self, uid: str) -> list[str]:
        async with self._lock:
            iid_index = self._uid_to_iid_first_index.get(uid, {})
            return [iid for iid, _ in sorted(iid_index.items(), key=lambda kv: kv[1])]

    async def version(self) -> int:
        async with self._lock:
            return self._version

    async def query(
        self,
        *,
        msg_type: str | None = None,
        uid: str | None = None,
        iid: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            results: Iterable[dict[str, Any]] = self._messages

            # Helper to extract inner body.type if present
            def get_inner_type(m: dict[str, Any]) -> str | None:
                body = m.get("body")
                if isinstance(body, dict):
                    t = body.get("type")
                    return t if isinstance(t, str) else None
                if isinstance(body, str):
                    try:
                        parsed = json.loads(body)
                        if isinstance(parsed, dict):
                            t = parsed.get("type")
                            return t if isinstance(t, str) else None
                    except Exception:
                        return None
                return None

            # Legacy single-type filter (matches outer type only)
            if msg_type:
                results = (m for m in results if m.get("type") == msg_type)

            # Include filter: keep if matches any include (outer or inner)
            if include:
                include_set = set(include)
                results = (
                    m
                    for m in results
                    if (m.get("type") in include_set) or (get_inner_type(m) in include_set)
                )

            # Exclude filter: drop if matches any exclude (outer or inner)
            if exclude:
                exclude_set = set(exclude)
                results = (
                    m
                    for m in results
                    if (m.get("type") not in exclude_set) and (get_inner_type(m) not in exclude_set)
                )
            if uid:
                results = (m for m in results if m.get("uid") == uid)
            if iid:
                results = (m for m in results if m.get("iid") == iid)
            # sort by timestamp if present
            out = list(results)
            try:
                out.sort(key=lambda m: m.get("timestamp") or "")
            except Exception:
                pass
            if limit is not None and limit >= 0:
                return out[-limit:]
            return out

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()
            self._types.clear()
            self._uid_first_index.clear()
            self._uid_to_iid_first_index.clear()
            # bump version so clients polling /version can refresh
            self._version += 1


STORE = LogStore()
