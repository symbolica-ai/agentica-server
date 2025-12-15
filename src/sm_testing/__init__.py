from os import environ

from agentica_internal.core.print import eprint, hprint, tprint

from .sandbox_tst_ctx import *

__all__ = [
    'SandboxTestContext',
    'Sandbox',
    'SDKWorld',
    'log_sm_tests',
    'hprint',
    'tprint',
    'eprint',
]


def log_sm_tests():
    environ['SM_TEST_LOGGING'] = '1'
