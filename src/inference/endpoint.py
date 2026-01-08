import json
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from application.http_client import get_client
from messages import InvocationNotifier, Notifier

logger = getLogger(__name__)


@dataclass
class InferenceEndpoint:
    inference_endpoint: str
    inference_token: str
    notifier: Notifier
    fresh_id: Callable[[], str]
    user_id: str | None = None

    def _get_server_info(self) -> tuple[str | None, int | None]:
        """Parse endpoint URL to extract server.address and server.port per OTel spec."""
        try:
            parsed = urlparse(self.inference_endpoint)
            host = parsed.hostname
            port = parsed.port
            # If no explicit port, use default based on scheme
            if port is None and parsed.scheme:
                port = 443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else None
            return (host, port)
        except Exception as e:
            logger.warning(f"Failed to parse inference endpoint URL: {e}")
            return (None, None)

    async def invoke_stream(
        self,
        json_dict: dict[str, Any],
        timeout: int | None = None,
        iid: str = "no-iid",
        invocation: InvocationNotifier | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Inference streaming via SSE: Server-Sent Events."""
        this_id = self.fresh_id()
        timeout_ = httpx.Timeout(timeout, read=None)
        DATA_PREFIX = "data:"
        END_MARKER = "[DONE]"

        client = get_client()
        # Set headers for client
        headers: dict[str, str] = {}
        if self.inference_token:
            headers["Authorization"] = f"Bearer {self.inference_token}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        # Stream the response
        async with client.stream(
            "POST",
            self.inference_endpoint,
            json=json_dict,
            headers=headers,
            timeout=timeout_,
        ) as res:
            try:
                server_address, server_port = self._get_server_info()
                if invocation:
                    invocation.start_inference(
                        inference_id=this_id,
                        request=json_dict,
                        streaming=True,
                        server_address=server_address,
                        server_port=server_port,
                    )
                    # Start the gen_ai.chat span BEFORE streaming begins
                    # This captures the full streaming duration
                    start_event = invocation.create_chat_event(
                        inference_id=this_id,
                        streaming=True,
                        server_address=server_address,
                        server_port=server_port,
                    )
                    await invocation.log_genai_chat(start_event)
                await self.notifier.on_inference_request(
                    inference_id=this_id,
                    iid=iid,
                    request_str=json.dumps(json_dict),
                    timeout=timeout,
                )
                res.raise_for_status()
                async for line in res.aiter_lines():
                    if not line or not line.startswith(DATA_PREFIX):
                        continue
                    data = line[len(DATA_PREFIX) :].strip()
                    if data == END_MARKER:
                        break
                    try:
                        await self.notifier.on_inference_response(
                            inference_id=this_id,
                            iid=iid,
                            response_str=data,
                        )
                        yield json.loads(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Inference streaming JSON decode error: {e}")
                        continue
            except BaseException as e:
                await self.notifier.on_inference_error(
                    inference_id=this_id,
                    iid=iid,
                    err=e,
                    message=traceback.format_exc(),
                )
                raise e

    async def invoke(
        self,
        json_dict: dict[str, Any],
        timeout: int | None = None,
        iid: str = "no-iid",
        invocation: InvocationNotifier | None = None,
    ) -> dict[str, Any]:
        """Inference via HTTP POST."""
        this_id = self.fresh_id()
        client = get_client()
        headers: dict[str, str] = {}
        if self.inference_token:
            headers["Authorization"] = f"Bearer {self.inference_token}"
        headers["Content-Type"] = "application/json"
        try:
            server_address, server_port = self._get_server_info()
            if invocation:
                invocation.start_inference(
                    inference_id=this_id,
                    request=json_dict,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                # Start the gen_ai.chat span BEFORE the LLM call
                # This captures the actual inference duration
                start_event = invocation.create_chat_event(
                    inference_id=this_id,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                await invocation.log_genai_chat(start_event)
            await self.notifier.on_inference_request(
                inference_id=this_id,
                iid=iid,
                request_str=json.dumps(json_dict),
                timeout=timeout,
            )
            res = await client.post(
                self.inference_endpoint,
                json=json_dict,
                headers=headers or None,
                timeout=timeout,
            )
            res.raise_for_status()
            try:
                result = res.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in inference response: {e}")
                raise e
            await self.notifier.on_inference_response(
                inference_id=this_id,
                iid=iid,
                response_str=json.dumps(result),
            )
            if invocation:
                server_address, server_port = self._get_server_info()
                event = invocation.create_chat_event(
                    inference_id=this_id,
                    response=result,
                    streaming=False,
                    server_address=server_address,
                    server_port=server_port,
                )
                await invocation.log_genai_chat(event)
            return result
        except BaseException as e:
            await self.notifier.on_inference_error(
                inference_id=this_id,
                iid=iid,
                err=e,
                message=traceback.format_exc(),
            )
            raise e

    def is_openrouter(self) -> bool:
        return self.inference_endpoint.startswith('https://openrouter.ai/api/v1')

    async def openrouter_models(self) -> list[str]:
        """ad-hoc method supported only for OpenRouter endpoints."""
        if not self.is_openrouter():
            raise TypeError("OpenRouter models are only supported for OpenRouter endpoints")
        models_endpoint = self.inference_endpoint.replace('/chat/completions', '/models/user')
        client = get_client()
        headers: dict[str, str] = {}
        if self.inference_token:
            headers["Authorization"] = f"Bearer {self.inference_token}"
        response = await client.get(
            models_endpoint,
            headers=headers,
        )
        response.raise_for_status()
        models = response.json()['data']

        text_models = [
            model['id']
            for model in models
            if 'text' in model.get('architecture', {}).get('input_modalities', [])
            and 'text' in model.get('architecture', {}).get('output_modalities', [])
        ]
        return text_models

    async def openrouter_model_exists(self, model_id: str) -> bool:
        """ad-hoc method supported only for OpenRouter endpoints."""
        if not self.is_openrouter():
            raise TypeError(
                "OpenRouter models are only supported for OpenRouter endpoints. "
                f"Got: `{self.inference_endpoint}`"
            )
        params_endpoint = self.inference_endpoint.replace('/chat/completions', '/parameters')

        parts = model_id.split('/', 1)
        if len(parts) != 2:
            return False

        author, slug = parts
        params_endpoint = f"{params_endpoint}/{author}/{slug}"

        client = get_client()
        headers: dict[str, str] = {}
        if self.inference_token:
            headers["Authorization"] = f"Bearer {self.inference_token}"
        response = await client.get(params_endpoint, headers=headers)
        return response.is_success

    async def authenticate(self) -> None:
        if not self.is_openrouter():
            return None
        key_endpoint = self.inference_endpoint.replace('/chat/completions', '/key')

        client = get_client()
        headers: dict[str, str] = {}
        if self.inference_token:
            headers["Authorization"] = f"Bearer {self.inference_token}"
        response = await client.get(key_endpoint, headers=headers)
        if response.status_code == 401:
            raise UnauthorizedInferenceEndpoint(
                "Invalid or missing API key",
                request=response.request,
                response=response,
            )
        response.raise_for_status()


class UnauthorizedInferenceEndpoint(httpx.HTTPStatusError):
    """Raised when the inference endpoint returns a 401 Unauthorized response."""

    pass
