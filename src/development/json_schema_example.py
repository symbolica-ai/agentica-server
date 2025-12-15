import asyncio
import os
from typing import Literal

from com.context_json import Exec
from com.utils import asyncify

from com.apis import API
from com.context import CallableExecutable, Context, GenContext, GenState
from com.functional import const
from com.monads import gen, insert, update_with_execute
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

    def make_person(name: str, age: int, gender: Literal['male', 'female']) -> Person:
        """Make a Person object."""
        return Person(name, age, gender)

    person_executable = CallableExecutable(
        name='make_person',
        obj=make_person,
        description='A function to make a Person object',
    )

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
        >> gen(callables=[make_person])
        >> update_with_execute
        >> insert("Make a person named Jane Doe.", name='user')
        >> gen(callables=[make_person])
        >> update_with_execute
    )
    # fmt: on

    async def main():
        await ctx.run(m)
        print(ctx)
        print(ctx.gen.usage)

    asyncio.run(main())
