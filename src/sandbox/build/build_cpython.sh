#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
echo "Step 1: Build CPython for WASI"

TOOL_DIR="$ROOT/_tooling"
mkdir -p "$TOOL_DIR"

WASI_VER="27.0"
TAG="wasi-sdk-27"

# Detect platform
ARCH=$(uname -m)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    # Normalize architecture name for wasi-sdk
    if [[ "$ARCH" == "aarch64" ]]; then
        ARCH="arm64"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi
PLATFORM="${ARCH}-${OS}"
echo "Platform Detected: $PLATFORM"

WASI_TARBALL="$TOOL_DIR/wasi-sdk-${WASI_VER}-${PLATFORM}.tar.gz"

if [ ! -f "$WASI_TARBALL" ]; then
  WASI_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/${TAG}/wasi-sdk-${WASI_VER}-${PLATFORM}.tar.gz"
  echo "[wasi-sdk] Downloading ${WASI_URL}"
  curl -fL -o "$WASI_TARBALL" "$WASI_URL"
else
  echo "[wasi-sdk] Using existing $WASI_TARBALL"
fi

echo "[wasi-sdk] Verifying checksum"
(cd "$TOOL_DIR" && shasum -a 256 -c "$ROOT/wasi-sdk.sha256sum" --ignore-missing)

WASI_DIR="$TOOL_DIR/wasi-sdk-${WASI_VER}"
if [ ! -d "$WASI_DIR" ]; then
  echo "[wasi-sdk] Extracting to $WASI_DIR"
  rm -rf "$WASI_DIR" || true
  tar -C "$TOOL_DIR" -xf "$WASI_TARBALL"
  mv "${WASI_DIR}-${PLATFORM}" $WASI_DIR
else
  echo "[wasi-sdk] Using existing $WASI_DIR"
fi

WASI_SDK="$WASI_DIR"
if [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "Error: clang not found at $WASI_SDK/bin/clang"
  exit 1
fi

CLANG_RES="$("$WASI_SDK/bin/clang" --print-resource-dir)"
RT_DIR="$CLANG_RES/lib/wasm32-unknown-wasi"
RT_BUILTINS="$RT_DIR/libclang_rt.builtins.a"
if [ ! -f "$RT_BUILTINS" ]; then
  echo "Error: compiler-rt builtins not found at: $RT_BUILTINS"
  echo "Ensure you're using the official wasi-sdk ${WASI_VER} macOS tarball."
  exit 1
fi

HEADERS_OUT="$TOOL_DIR/py-headers-3.12"
if [ ! -d "$HEADERS_OUT" ]; then
    PY_VER="3.12.0"
    PY_TARBALL="$TOOL_DIR/Python-${PY_VER}.tgz"
    PY_SRC="$TOOL_DIR/Python-${PY_VER}"
    PY_PREFIX="$PY_SRC/build-wasi/install"

    if [ ! -f "$PY_TARBALL" ]; then
    echo "[cpython] Downloading CPython ${PY_VER}"
    curl -fL -o "$PY_TARBALL" "https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz"
    else
    echo "[cpython] Using existing $PY_TARBALL"
    fi

    echo "[cpython] Verifying checksum"
    (cd "$TOOL_DIR" && shasum -a 256 -c "$ROOT/python.sha256sum")

    if [ ! -d "$PY_SRC" ]; then
    echo "[cpython] Extracting to $PY_SRC"
    rm -rf "$PY_SRC" || true
    tar -C "$TOOL_DIR" -xf "$PY_TARBALL"
    else
    echo "[cpython] Using existing $PY_SRC"
    fi

    pushd "$PY_SRC" >/dev/null

    export WASI_SDK_PATH="$WASI_SDK"
    export CONFIG_SITE="$PWD/Tools/wasm/wasm-config.site"

    # this should only install python, there's nothing else in this pyproject file and nothing depends on build
    uv sync
    BUILD_PY="$(uv run python -c 'import sys; print(sys.executable)')"

    echo "[cpython] Using BUILD_PY: $BUILD_PY"

    mkdir -p build-wasi

    MERGED_SITE="$PWD/build-wasi/config.site"
    {
    if [ -f "$PWD/Tools/wasm/wasm-config.site" ]; then
        echo ". \"$PWD/Tools/wasm/wasm-config.site\""
    fi
    echo "ac_cv_file__dev_ptmx=no"
    echo "ac_cv_file__dev_ptc=no"
    } > "$MERGED_SITE"
    export CONFIG_SITE="$MERGED_SITE"

    echo "[cpython] Configuring with CONFIG_SITE: $CONFIG_SITE"

    CC="$WASI_SDK/bin/clang --target=wasm32-wasi --sysroot=$WASI_SDK/share/wasi-sysroot" \
    AR="$WASI_SDK/bin/llvm-ar" \
    RANLIB="$WASI_SDK/bin/llvm-ranlib" \
    READELF="$WASI_SDK/bin/llvm-readelf" \
    STRIP="$WASI_SDK/bin/llvm-strip" \
    CFLAGS="--sysroot=$WASI_SDK/share/wasi-sysroot -O2" \
    LDFLAGS="-L$RT_DIR" \
    ./Tools/wasm/wasi-env ./configure \
    --host=wasm32-unknown-wasi \
    --build="$(./config.guess)" \
    --with-build-python="$BUILD_PY" \
    --prefix="$PY_PREFIX"

    echo "[cpython] Copying headers to $TOOL_DIR/py-headers-3.12"

    HEADERS_OUT="$TOOL_DIR/py-headers-3.12"
    rm -rf "$HEADERS_OUT" || true
    mkdir -p "$HEADERS_OUT"
    cp -R "$PWD/Include/"* "$HEADERS_OUT/"
    cp "$PWD/pyconfig.h" "$HEADERS_OUT/"
    popd >/dev/null

    echo "[cpython] PY_INC=$HEADERS_OUT"
    echo "[cpython] PY_ABI=cp312"
fi
