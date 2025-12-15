import asyncio
import os
from typing import Literal

from com.context_json import Exec, Executable
from com.utils import asyncify

from com.apis import API
from com.context import Context, GenContext, GenState
from com.functional import const
from com.monads import gen, insert, update
from inference.endpoint import InferenceEndpoint
from sandbox import Sandbox

if __name__ == "__main__":
    key = os.getenv('OPENAI_API_KEY')
    assert key is not None

    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int
        gender: Literal['male', 'female']

    person_executable = Executable(name='Person', description='A person', id='abcd', obj=Person)

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
            executables={person_executable.name: person_executable},
            sandbox=Sandbox(
                warp_send_bytes=asyncify(const(None)), warp_recv_bytes=asyncify(const(b""))
            ),
        ),
        state=GenState.TEXT,
    )

    # fmt: off
    m = (
        insert("Make a person named John Doe, 31 years old, who is non-binary.", name='user')
        >> insert("I have made the following person:\n", name='agent')
        >> gen(constraints=[Person])
        >> update
    )
    # fmt: on

    async def main():
        await ctx.run(m)
        print(ctx)

    asyncio.run(main())
