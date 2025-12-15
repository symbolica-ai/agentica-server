import re
from pathlib import Path
from warnings import warn

THIS_FILE = Path(__file__)
TEST_PATH = THIS_FILE.parent

FORBIDDEN_RE = re.compile(r'\b(Sandbox|SDKWorld)\(')


def test_test_context_usage():
    for file in TEST_PATH.rglob('*.py'):
        name = file.name
        if file == THIS_FILE or name == 'conftest.py' or 'broken_' in name:
            continue
        if name == 'test_converter_integration.py':
            # skip for now...
            continue
        source = file.read_text()
        matches = re.findall(FORBIDDEN_RE, source)
        if len(matches):
            warn(f"'{file}' uses Sandbox() or SDKWorld(); use SandboxTestContext")
