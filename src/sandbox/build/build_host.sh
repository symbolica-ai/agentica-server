#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Step 4: Build Rust library"
cd "$ROOT"/../host

rm env.wasm.compiled || true
