import asyncio
import os
from typing import Literal

from com.context_json import Exec
from com.utils import asyncify

from com.abstract import HistoryMonad
from com.apis import API
from com.context import Context, GenContext, GeneratedDelta, GenState, ObjectExecutable
from com.do import do
from com.functional import const
from com.monads import gen, pure, update
from inference import InferenceEndpoint
from sandbox import Sandbox


def fresh_context() -> Context:
    key = os.getenv('OPENAI_API_KEY')
    assert key is not None

    return Context(
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


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int
        gender: Literal['male', 'female']

    person_executable = ObjectExecutable(name='Person', obj=Person, description='A person')

    @do(HistoryMonad[str])
    def get_name():
        yield insert_string(
            "Make a person named John Doe, 31 years old, who is non-binary.", name='user'
        )
        res: GeneratedDelta = yield gen(objects=['Person'])
        yield update(res)
        print("the generated delta:", res.content, end='\n\n')
        yield pure(res.content)

    m: HistoryMonad[str] = get_name()

    async def main():
        ctx = fresh_context()
        x = await ctx.run(m)
        print("[1] name is:", x)
        # rerun same monad in fresh context
        ctx = fresh_context()
        x = await ctx.run(m)
        print("[2] name is:", x)

    asyncio.run(main())
