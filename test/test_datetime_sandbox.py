import pytest


@pytest.mark.asyncio
async def test_exec(make_dummy_sandbox):
    async with make_dummy_sandbox(
        logging=False,
    ) as sb:
        out, stdout, stderr = await sb.repl_exec("""
import datetime        
date1 = datetime.datetime.strptime('30 October 2025', '%d %B %Y').date()
date2 = datetime.datetime.strptime('15 Jan 2024', '%d %b %Y').date()
date3 = datetime.datetime.strptime('2024-03-20', '%Y-%m-%d').date()
print(f"Parsed dates: {date1}, {date2}, {date3}", end='')
""")
        assert out == "Parsed dates: 2025-10-30, 2024-01-15, 2024-03-20"
        assert stdout == "Parsed dates: 2025-10-30, 2024-01-15, 2024-03-20"
        assert stderr == ""
