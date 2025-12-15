import asyncio
import logging

from com.context_json import Exec
from com.utils import asyncify

from com.apis import API
from com.context import Context, GenContext, GenState
from com.functional import const
from com.monads import gen, insert, update
from inference.endpoint import InferenceEndpoint
from inference.mock.endpoint import CompletionsEndpoint, Reply
from sandbox import Sandbox

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    mock = CompletionsEndpoint()
    mock.start_threaded()

    ctx = Context(
        gen=GenContext(
            model='gpt-4o',
            api=API.OPENAI_CHAT_COMPLETIONS,
            deltas=[],
            max_completion_tokens=1000,
            type='json',
            guided=True,
            endpoint=InferenceEndpoint(
                inference_endpoint=mock.endpoint_url,
                inference_token='None',
            ),
        ),
        exec=Exec(
            executables={},
            sandbox=Sandbox(resource_callback=asyncify(const(b""))),
        ),
        state=GenState.TEXT,
    )

    # fmt: off
    m = (
        insert("Tell me a joke.", name='user')
        >> gen()
        >> update
    )
    # fmt: on

    async def main():
        done = ctx.run(m)  # start running
        # run the monad, will hang on the generation
        # until we add a response to the response queue
        mock.respond(
            "/v1/chat/completions",
            Reply(
                "Uh, okay, so... um, have you ever, like... you know when you're at the supermarket and, [...]"
            ),
        )
        await done
        print(ctx)

    asyncio.run(main())
