from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litestar import Litestar, Request, get, post
from litestar.response import Response
from litestar.static_files import create_static_files_router

# === In-memory state ===


@dataclass
class PendingRequest:
    request_id: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=lambda: time.time())
    # Result is (status_code, JSON body)
    future: asyncio.Future[tuple[int, Dict[str, Any]]] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )
    completed: bool = False
    status: Optional[int] = None


class PendingStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending_by_id: Dict[str, PendingRequest] = {}
        self._sent: List[Dict[str, Any]] = []
        self._last_completed_id: Optional[str] = None

    async def add(self, payload: Dict[str, Any]) -> PendingRequest:
        request_id = str(uuid.uuid4())
        pr = PendingRequest(request_id=request_id, payload=payload)
        async with self._lock:
            # If we have a lingering completed request, clear it now
            if self._last_completed_id:
                self._pending_by_id.pop(self._last_completed_id, None)
                self._last_completed_id = None
            self._pending_by_id[request_id] = pr
        return pr

    async def get(self, request_id: str) -> Optional[PendingRequest]:
        async with self._lock:
            return self._pending_by_id.get(request_id)

    async def remove(self, request_id: str) -> None:
        async with self._lock:
            self._pending_by_id.pop(request_id, None)

    async def list_summaries(self) -> List[Dict[str, Any]]:
        async with self._lock:
            # Return newest first
            items = sorted(self._pending_by_id.values(), key=lambda p: p.created_at, reverse=True)
            summaries: List[Dict[str, Any]] = []
            for p in items:
                messages = p.payload.get("messages", [])
                model = p.payload.get("model")
                stop = p.payload.get("stop")
                max_tokens = p.payload.get("max_completion_tokens") or p.payload.get("max_tokens")
                summaries.append(
                    {
                        "request_id": p.request_id,
                        "created_at": p.created_at,
                        "model": model,
                        "messages": messages,
                        "stop": stop,
                        "max_completion_tokens": max_tokens,
                        "completed": p.completed,
                        "status": p.status,
                    }
                )
            return summaries

    async def add_sent(self, entry: Dict[str, Any]) -> None:
        async with self._lock:
            # Normalize to include created_at and avoid double-wrapping
            normalized = {"created_at": time.time(), **entry}
            self._sent.append(normalized)

    async def list_sent(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._sent)

    async def mark_completed(self, request_id: str, status: int) -> None:
        async with self._lock:
            pr = self._pending_by_id.get(request_id)
            if pr is not None:
                pr.completed = True
                pr.status = status
                # Remember this completed request so we can evict it on the next add()
                self._last_completed_id = request_id


pending_store = PendingStore()


# === Helpers ===


def _truncate_with_stop(content: str, stop: Optional[Any]) -> tuple[str, Optional[str]]:
    if not stop:
        return content, None
    stop_list = [stop] if isinstance(stop, str) else list(stop)
    earliest_index: Optional[int] = None
    found_stop: Optional[str] = None
    for s in stop_list:
        idx = content.find(s)
        if idx != -1 and (earliest_index is None or idx < earliest_index):
            earliest_index = idx
            found_stop = s
    if earliest_index is None:
        return content, None
    return content[:earliest_index], found_stop


def _truncate_by_max_tokens(content: str, max_tokens: Optional[int]) -> tuple[str, bool]:
    if not max_tokens:
        return content, False
    # Naive character-based fallback when no tokenizer is available.
    if len(content) <= max_tokens:
        return content, False
    return content[:max_tokens], True


def _estimate_usage(messages: List[Dict[str, Any]], completion_text: str) -> Dict[str, int]:
    # Heuristic token estimate: 1 token ~= 4 chars. This is only for debugging UI.
    def approx_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    prompt_text = "\n".join(
        [
            (m.get("content") or "")
            + ("\n" + (m.get("reasoning_content") or "") if m.get("reasoning_content") else "")
            for m in messages
        ]
    )
    prompt_tokens = approx_tokens(prompt_text)
    completion_tokens = approx_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# === OpenAI-like endpoint ===


@post(path="/openai/v1/chat/completions")
async def create_chat_completion(request: Request) -> Response[Dict[str, Any]]:
    payload = await request.json()
    pr = await pending_store.add(payload)

    # Wait for human-provided response from UI
    status, body = await pr.future

    # Mark as completed and keep it lingering until next add()
    await pending_store.mark_completed(pr.request_id, status)
    return Response(body, status_code=status)


# === UI APIs ===


@get(path="/mock/pending")
async def get_pending() -> List[Dict[str, Any]]:
    return await pending_store.list_summaries()


@post(path="/mock/respond")
async def post_response(request: Request) -> Dict[str, Any]:
    body = await request.json()
    request_id: str = body["request_id"]
    content: str = body.get("content", "")
    status: int = int(body.get("status", 200))

    pr = await pending_store.get(request_id)
    if pr is None:
        return {"ok": False, "error": "request_id not found"}

    messages = pr.payload.get("messages", [])
    model = pr.payload.get("model")
    stop = pr.payload.get("stop")
    max_tokens = pr.payload.get("max_completion_tokens") or pr.payload.get("max_tokens")

    if status == 200:
        # Apply stop tokens first, then max token truncation
        truncated, stop_text = _truncate_with_stop(content, stop)
        truncated, hit_length = _truncate_by_max_tokens(truncated, max_tokens)

        finish_reason = "stop" if stop_text is not None else ("length" if hit_length else "stop")

        usage = _estimate_usage(messages, truncated)

        # Build OpenAI-like response
        now = int(time.time())
        chat_completion: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": truncated,
                        "annotations": None,
                        "refusal": None,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

        if not pr.future.done():
            pr.future.set_result((200, chat_completion))
            await pending_store.add_sent(
                {
                    "id": f"sent-{uuid.uuid4().hex[:12]}",
                    "status": 200,
                    "model": model,
                    "request_id": request_id,
                    "response": chat_completion,
                }
            )
        return {"ok": True}
    else:
        error_body: Dict[str, Any] = {
            "error": {
                "message": content or f"Mock error {status}",
                "type": "mock",
                "code": status,
            }
        }
        if not pr.future.done():
            pr.future.set_result((status, error_body))
            await pending_store.add_sent(
                {
                    "id": f"sent-{uuid.uuid4().hex[:12]}",
                    "status": status,
                    "model": model,
                    "request_id": request_id,
                    "response": error_body,
                }
            )
        # For non-200 statuses, remove immediately from pending
        await pending_store.remove(pr.request_id)
        return {"ok": True}


@get(path="/mock/sent")
async def get_sent() -> List[Dict[str, Any]]:
    return await pending_store.list_sent()


@get(path="/mock/pending/{request_id:str}")
async def get_pending_request(request_id: str) -> Dict[str, Any]:
    pr = await pending_store.get(request_id)
    if pr is None:
        return {"error": "not_found"}
    return pr.payload


# === App ===


static_router = create_static_files_router(
    path="/mock/ui",
    directories=[__file__.rsplit("/", 1)[0] + "/web"],
    html_mode=True,
)

app = Litestar(
    route_handlers=[
        create_chat_completion,
        get_pending,
        post_response,
        get_sent,
        get_pending_request,
        static_router,
    ]
)


if __name__ == "__main__":
    # For manual runs: python -m inference.mock.server
    import uvicorn

    uvicorn.run("development.mock_endpoint.server:app", host="0.0.0.0", port=8000, reload=True)
