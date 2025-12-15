#!/bin/bash
set -euo pipefail

# Builds python-xxhash's C extension for WASI and packages a tarball suitable for
# extraction into a Python site-packages/ directory.
# Output: _tooling/xxhash-wasi.tar.gz

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TOOL_DIR="$ROOT/_tooling"
mkdir -p "$TOOL_DIR" || true

# Configuration
# Optionally set XXHASH_VER env var to pin; defaults to 3.4.1
XXHASH_VER="${XXHASH_VER:-3.4.1}"
OUT_TAR="$TOOL_DIR/xxhash-wasi.tar.gz"
BUILD_DIR="$TOOL_DIR/build-xxhash"
PKG_DIR="$BUILD_DIR/pkg"

# Cache: if the output already exists, skip the build
if [ -f "$OUT_TAR" ]; then
  echo "[xxhash] Cache hit: $OUT_TAR already exists; skipping build"
  exit 0
fi

# Ensure WASI SDK and Python headers are available (prepared by build_cpython.sh)
WASI_VER="27.0"
WASI_DIR="$TOOL_DIR/wasi-sdk-${WASI_VER}"
WASI_SDK="$WASI_DIR"
if [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "[xxhash] Error: WASI SDK not found; run build_cpython.sh first"
  exit 1
fi

# Python headers (produced by build_cpython.sh) - only CPython 3.12 for now
PY_HEADERS_312_DIR="$TOOL_DIR/py-headers-3.12"
if [ -d "$PY_HEADERS_312_DIR" ]; then
  PY_HEADERS_DIR="$PY_HEADERS_312_DIR"
  PY_ABI_TAG="cp312"
else
  echo "[xxhash] Error: Python 3.12 headers not found; run build_cpython.sh first"
  exit 1
fi

# Map to cpython tag used in extension filenames (3.12 only)
CPYTHON_TAG="cpython-312"

# Resolve compiler rt dir for linking
CLANG_RES="$("$WASI_SDK/bin/clang" --print-resource-dir)"
RT_DIR="$CLANG_RES/lib/wasm32-unknown-wasi"
if [ ! -d "$RT_DIR" ]; then
  echo "[xxhash] Error: compiler runtime dir not found: $RT_DIR"
  exit 1
fi

rm -rf "$BUILD_DIR" || true
mkdir -p "$BUILD_DIR" "$PKG_DIR" || true

# Download python-xxhash source (sdist from PyPI)
ARCHIVE_URL="https://files.pythonhosted.org/packages/source/x/xxhash/xxhash-${XXHASH_VER}.tar.gz"
ARCHIVE_FILE="$TOOL_DIR/xxhash-${XXHASH_VER}-pypi.tar.gz"

if [ ! -f "$ARCHIVE_FILE" ]; then
  echo "[xxhash] Downloading ${ARCHIVE_URL}"
  curl -fL -o "$ARCHIVE_FILE" "$ARCHIVE_URL"
else
  echo "[xxhash] Using cached $ARCHIVE_FILE"
fi

echo "[xxhash] Verifying checksum"
(cd "$TOOL_DIR" && shasum -a 256 -c "$ROOT/xxhash.sha256sum")

echo "[xxhash] Unpacking $ARCHIVE_FILE"
tar -C "$BUILD_DIR" -xf "$ARCHIVE_FILE"

# The sdist extracts to xxhash-<version>/
SRC_TOP_DIR="$BUILD_DIR/xxhash-${XXHASH_VER}"
if [ ! -d "$SRC_TOP_DIR" ]; then
  echo "[xxhash] Error: Unexpected sdist layout; cannot find $SRC_TOP_DIR"
  exit 1
fi

# Copy pure-Python package files
mkdir -p "$PKG_DIR/xxhash" || true
cp -R "$SRC_TOP_DIR/xxhash/"*.py "$PKG_DIR/xxhash/"
cp -R "$SRC_TOP_DIR/xxhash/"*.pyi "$PKG_DIR/xxhash/" 2>/dev/null || true
if [ -f "$SRC_TOP_DIR/xxhash/py.typed" ]; then
  cp "$SRC_TOP_DIR/xxhash/py.typed" "$PKG_DIR/xxhash/"
fi

echo "[xxhash] Compiling C extension for WASI"
OBJ_DIR="$BUILD_DIR/obj"
mkdir -p "$OBJ_DIR" || true

# Determine C source file and resulting extension basename
EXT_BASENAME="_xxhash"

WRAP_C=""

# Prefer new sdist layout under src/_xxhash.c; fallback to legacy xxhash/_xxhash.c
if [ -f "$SRC_TOP_DIR/src/_xxhash.c" ]; then
  WRAP_C="$SRC_TOP_DIR/src/_xxhash.c"
elif [ -f "$SRC_TOP_DIR/xxhash/_xxhash.c" ]; then
  WRAP_C="$SRC_TOP_DIR/xxhash/_xxhash.c"
else
  echo "[xxhash] Error: Could not find Python wrapper source (_xxhash.c) in src/ or xxhash/"
  exit 1
fi

# Compile
# Include the source directory for any local headers, plus bundled deps/xxhash
INC_DEPS="-I$SRC_TOP_DIR/src -I$SRC_TOP_DIR/xxhash -I$SRC_TOP_DIR/deps/xxhash"

"$WASI_SDK/bin/clang" \
  --target=wasm32-wasi \
  --sysroot="$WASI_SDK/share/wasi-sysroot" \
  -I"$PY_HEADERS_DIR" \
  -I"$PY_HEADERS_DIR/cpython" \
  $INC_DEPS \
  -O3 -DNDEBUG -fPIC -std=c99 -fvisibility=default \
  -DSSIZE_MAX=2147483647 \
  -c "$WRAP_C" \
  -o "$OBJ_DIR/${EXT_BASENAME}.o"

# Build the bundled xxhash core implementation
"$WASI_SDK/bin/clang" \
  --target=wasm32-wasi \
  --sysroot="$WASI_SDK/share/wasi-sysroot" \
  $INC_DEPS \
  -O3 -DNDEBUG -fPIC -std=c99 -fvisibility=default \
  -DSSIZE_MAX=2147483647 \
  -c "$SRC_TOP_DIR/deps/xxhash/xxhash.c" \
  -o "$OBJ_DIR/xxhash_core.o"

# Link shared object for Python extension using wasm-ld.
EXT_NAME="${EXT_BASENAME}.${CPYTHON_TAG}-wasm32-wasi.so"
"$WASI_SDK/bin/wasm-ld" \
  --no-entry --shared --export-dynamic --allow-undefined --export=PyInit__xxhash \
  -L"$RT_DIR" \
  -o "$PKG_DIR/xxhash/$EXT_NAME" \
  "$OBJ_DIR/${EXT_BASENAME}.o" \
  "$OBJ_DIR/xxhash_core.o" \
  "$RT_DIR/libclang_rt.builtins.a"

echo "[xxhash] Built: $PKG_DIR/xxhash/$EXT_NAME"

echo "[xxhash] Creating tarball: $OUT_TAR"
rm -f "$OUT_TAR"
tar -C "$PKG_DIR" -czf "$OUT_TAR" xxhash

echo "[xxhash] Done"