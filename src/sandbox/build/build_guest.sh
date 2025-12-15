#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
( set +x; echo "Step 2: Stage deps (deps/site-packages/...)" )
DEPS_DIR="$ROOT/../guest/.venv/lib/python3.12/site-packages/"
TOOL_DIR="$ROOT/_tooling"
mkdir "$TOOL_DIR" || true

# Ensure WASI SDK and Python headers exist; trigger CPython build if missing
WASI_VER="27.0"
WASI_DIR="$TOOL_DIR/wasi-sdk-${WASI_VER}"
NEED_CPYTHON=0
if [ ! -x "$WASI_DIR/bin/clang" ]; then
    NEED_CPYTHON=1
fi
if [ ! -d "$TOOL_DIR/py-headers-3.12" ]; then
    NEED_CPYTHON=1
fi
if [ "$NEED_CPYTHON" -eq 1 ]; then
    ( set +x; echo "[guest] Building CPython/WASI toolchain via build_cpython.sh" )
    bash "$ROOT/build_cpython.sh"
fi

PYDANTIC_TAR="$TOOL_DIR/pydantic_core-wasi.tar.gz"
if [ ! -f "$PYDANTIC_TAR" ]; then
    echo "Dowloading $PYDANTIC_TAR"
    PYDANTIC_WASI_WHEEL=https://github.com/dicej/wasi-wheels/releases/download/latest/pydantic_core-wasi.tar.gz
    curl -fL "$PYDANTIC_WASI_WHEEL" -o "$PYDANTIC_TAR"
else
    echo "Using cached $PYDANTIC_TAR"
fi

echo "Verifying checksum"
(cd "$TOOL_DIR" && shasum -a 256 -c "$ROOT/pydantic.sha256sum")

# Build msgspec for WASI if not already present
MSGSPEC_TAR="$TOOL_DIR/msgspec-wasi.tar.gz"
if [ ! -f "$MSGSPEC_TAR" ]; then
    echo "[guest] Building msgspec-wasi.tar.gz via build_msgspec.sh"
    bash "$ROOT/build_msgspec.sh"
else
    echo "Using cached $MSGSPEC_TAR"
fi

# Build xxhash for WASI if not already present
XXHASH_TAR="$TOOL_DIR/xxhash-wasi.tar.gz"
if [ ! -f "$XXHASH_TAR" ]; then
    echo "[guest] Building xxhash-wasi.tar.gz via build_xxhash.sh"
    bash "$ROOT/build_xxhash.sh"
else
    echo "Using cached $XXHASH_TAR"
fi

( set +x; echo "Step 3: Componentize" )
cd "$ROOT"/../guest

rm -rf .venv

# uv cache clean # TODO: this invalidates too many venvs?
uv sync
source .venv/bin/activate
tar -xf $PYDANTIC_TAR -C $DEPS_DIR
tar -xf $MSGSPEC_TAR -C $DEPS_DIR
tar -xf $XXHASH_TAR -C $DEPS_DIR

# Make the_prelude.py
uv run python generate_prelude.py

cd ..

componentize-py \
  -d "$ROOT"/../wit \
  -w env \
  componentize \
  guest.execenv \
  -o "$ROOT/../env.wasm"
