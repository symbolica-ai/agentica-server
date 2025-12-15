import asyncio
import os
from textwrap import dedent

from com.context_json import Exec
from com.utils import asyncify

from agentic.monads import multi_turn_repl_monad
from com.apis import API
from com.context import Context, GenContext, GenState
from com.functional import const
from inference.endpoint import InferenceEndpoint
from sandbox import Sandbox

m = multi_turn_repl_monad(
    dedent("""
    Please return your current Python minor version number multiplied by 100.
""").strip()
)

if __name__ == "__main__":
    key = os.getenv('OPENAI_API_KEY')
    assert key is not None

    ctx = Context(
        gen=GenContext(
            model='gpt-4o',
            api=API.OPENAI_CHAT_COMPLETIONS,
            deltas=[],
            max_completion_tokens=1000,
            type='json',
            guided=True,
            endpoint=InferenceEndpoint(
                inference_endpoint='https://api.openai.com/v1/chat/completions',
                inference_token=key,
            ),
        ),
        exec=Exec(
            executables={},
            sandbox=Sandbox(
                warp_send_bytes=asyncify(const(None)), warp_recv_bytes=asyncify(const(b""))
            ),
        ),
        state=GenState.TEXT,
    )

    async def main():
        await ctx.run(m)
        print(ctx)

    asyncio.run(main())
