import asyncio

import pytest
from agentica_internal.repl.repl_eval_info import ReplEvaluationInfo
from agentica_internal.repl.repl_var_info import ReplVarInfo

from sm_testing import SandboxTestContext


@pytest.mark.asyncio
async def test_sandbox_accepts_future():
    my_future = asyncio.Future()
    context = SandboxTestContext('test_sandbox_accepts_future', sb_mode='no_sandbox')
    context.add_locals(my_future=my_future)

    async with context as sandbox:
        var_info: ReplVarInfo = await sandbox.repl_var_info('my_future')
        assert var_info is not None and var_info.kind == 'future'

        my_future.set_result(99)
        eval_info: ReplEvaluationInfo = await sandbox.repl_run_code("await my_future")
        assert eval_info.exception_name is None
        assert eval_info.traceback_str is None
        assert eval_info.out_str == '99'
