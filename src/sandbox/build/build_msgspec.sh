#!/bin/bash
set -euo pipefail

# Builds msgspec's C extension for WASI and packages a tarball suitable for
# extraction into a Python site-packages/ directory.
# Output: _tooling/msgspec-wasi.tar.gz

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TOOL_DIR="$ROOT/_tooling"
mkdir -p "$TOOL_DIR" || true

# Configuration
# Optionally set MSGSPEC_VER env var to pin; defaults to 0.19.0
MSGSPEC_VER="${MSGSPEC_VER:-0.19.0}"
OUT_TAR="$TOOL_DIR/msgspec-wasi.tar.gz"
BUILD_DIR="$TOOL_DIR/build-msgspec"
PKG_DIR="$BUILD_DIR/pkg"

# Cache: if the output already exists, skip the build
if [ -f "$OUT_TAR" ]; then
  echo "[msgspec] Cache hit: $OUT_TAR already exists; skipping build"
  exit 0
fi

# Ensure WASI SDK and Python headers are available (prepared by build_cpython.sh)
WASI_VER="27.0"
WASI_DIR="$TOOL_DIR/wasi-sdk-${WASI_VER}"
WASI_SDK="$WASI_DIR"
if [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "[msgspec] Error: WASI SDK not found; run build_cpython.sh first"
  exit 1
fi

# Python headers (produced by build_cpython.sh) - only CPython 3.12
PY_HEADERS_312_DIR="$TOOL_DIR/py-headers-3.12"
if [ -d "$PY_HEADERS_312_DIR" ]; then
  PY_HEADERS_DIR="$PY_HEADERS_312_DIR"
  PY_ABI_TAG="cp312"
else
  echo "[msgspec] Error: Python 3.12 headers not found; run build_cpython.sh first"
  exit 1
fi

# Map to cpython tag used in extension filenames (3.12 only)
CPYTHON_TAG="cpython-312"

# Resolve compiler rt dir for linking
CLANG_RES="$("$WASI_SDK/bin/clang" --print-resource-dir)"
RT_DIR="$CLANG_RES/lib/wasm32-unknown-wasi"
if [ ! -d "$RT_DIR" ]; then
  echo "[msgspec] Error: compiler runtime dir not found: $RT_DIR"
  exit 1
fi

rm -rf "$BUILD_DIR" || true
mkdir -p "$BUILD_DIR" "$PKG_DIR" || true

ARCHIVE_URL="https://github.com/jcrist/msgspec/archive/refs/tags/${MSGSPEC_VER}.tar.gz"
ARCHIVE_FILE="$TOOL_DIR/msgspec-${MSGSPEC_VER}-github.tar.gz"

if [ ! -f "$ARCHIVE_FILE" ]; then
  echo "[msgspec] Downloading ${ARCHIVE_URL}"
  curl -fL -o "$ARCHIVE_FILE" "$ARCHIVE_URL"
else
  echo "[msgspec] Using cached $ARCHIVE_FILE"
fi

echo "[msgspec] Verifying checksum"
(cd "$TOOL_DIR" && shasum -a 256 -c "$ROOT/msgspec.sha256sum")

echo "[msgspec] Unpacking $ARCHIVE_FILE"
tar -C "$BUILD_DIR" -xf "$ARCHIVE_FILE"

# The GitHub archive extracts to msgspec-<version>/
SRC_TOP_DIR="$BUILD_DIR/msgspec-${MSGSPEC_VER}"
if [ ! -d "$SRC_TOP_DIR" ]; then
  echo "[msgspec] Error: Unexpected sdist layout; cannot find $SRC_TOP_DIR"
  exit 1
fi

# Copy pure-Python package files
mkdir -p "$PKG_DIR/msgspec" || true
cp -R "$SRC_TOP_DIR/msgspec/"*.py "$PKG_DIR/msgspec/" || true
cp -R "$SRC_TOP_DIR/msgspec/"*.pyi "$PKG_DIR/msgspec/" || true
cp -R "$SRC_TOP_DIR/msgspec/py.typed" "$PKG_DIR/msgspec/" || true

echo "[msgspec] Compiling C extension (_core) for WASI"
OBJ_DIR="$BUILD_DIR/obj"
mkdir -p "$OBJ_DIR" || true

# Compile
"$WASI_SDK/bin/clang" \
  --target=wasm32-wasi \
  --sysroot="$WASI_SDK/share/wasi-sysroot" \
  -I"$PY_HEADERS_DIR" \
  -I"$PY_HEADERS_DIR/cpython" \
  -O3 -DNDEBUG -fPIC -std=c99 \
  -DSSIZE_MAX=2147483647 \
  -c "$SRC_TOP_DIR/msgspec/_core.c" \
  -o "$OBJ_DIR/_core.o"

# Link shared object for Python extension using wasm-ld.
EXT_NAME="_core.${CPYTHON_TAG}-wasm32-wasi.so"
"$WASI_SDK/bin/wasm-ld" \
  --no-entry --shared --export-dynamic --allow-undefined \
  -L"$RT_DIR" \
  -o "$PKG_DIR/msgspec/$EXT_NAME" \
  "$OBJ_DIR/_core.o" \
  "$RT_DIR/libclang_rt.builtins.a"

echo "[msgspec] Built: $PKG_DIR/msgspec/$EXT_NAME"

echo "[msgspec] Creating tarball: $OUT_TAR"
rm -f "$OUT_TAR" || true
tar -C "$PKG_DIR" -czf "$OUT_TAR" msgspec

echo "[msgspec] Done"