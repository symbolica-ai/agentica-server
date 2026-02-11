#!/bin/bash

set -e

cd "$(dirname "$(readlink -f "$0")")" || exit

BLUE='\033[0;34m'
BLUE_BG='\033[1;30;44m'
NC='\033[0m' # No Color

# prevents harmless 'VIRTUAL_ENV=XXX does not match the project environment path'
# when running this file from a different dir
unset VIRTUAL_ENV

echo -e "${BLUE_BG} LOCAL MODE TESTS ${NC}"
LOCAL_TESTING=1 SKIP_WASM=1 SKIP_LOCAL=0 uv run pytest -v -s --no-cov --log-disable=all --session-timeout 60 test "$@"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo
echo
echo -e "${BLUE_BG} WASM MODE TESTS ${NC}"
LOCAL_TESTING=1 SKIP_WASM=0 SKIP_LOCAL=1 uv run pytest -v -s --no-cov --log-disable=all --session-timeout 60 test "$@"
