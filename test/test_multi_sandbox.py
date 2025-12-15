import pytest

from sm_testing import *


@pytest.mark.asyncio
async def test_multiple_sandboxes():
    """Test that making different sandboxes with overlapping lifetimes in local mode do not influence one-another"""

    async with SandboxTestContext('sm1') as sb:
        await sb.repl_command("foo = 'OK'")
        names = await sb.repl_dir_vars('locals')
        assert "foo" in names

    async with SandboxTestContext('sm2') as sb:
        names = await sb.repl_dir_vars('locals')
        assert "foo" not in names
