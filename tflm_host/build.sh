#!/usr/bin/env bash
# Build and run the TensorFlow Lite Micro PC check. Run from Git Bash in the project folder:
#   bash tflm_host/build.sh          # 32-bit build (pointer size like the nRF52840) - default
#   bash tflm_host/build.sh 64       # 64-bit build
# Needs: tflm_host/.venv-tools (pip: ziglang cmake ninja), third_party/ (TFLM + downloads),
#        tflm_host/vectors/ (from export_vectors.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BITS="${1:-32}"
case "$BITS" in
  32) TARGET=x86-windows-gnu ;;
  64) TARGET=x86_64-windows-gnu ;;
  *) echo "usage: build.sh [32|64]"; exit 1 ;;
esac

TOOLS="$HERE/.venv-tools/Scripts"
ZIG="$("$TOOLS/python.exe" -c 'import ziglang, os; print(os.path.join(os.path.dirname(ziglang.__file__), "zig.exe"))')"
ZIG="$(cygpath -m "$ZIG")"

# Small wrappers so CMake/Ninja see ordinary compiler and archiver programs.
WRAP="$HERE/build/wrappers-$BITS"
mkdir -p "$WRAP"
printf '@"%s" cc -target %s %%*\r\n' "$ZIG" "$TARGET" > "$WRAP/zcc.cmd"
printf '@"%s" c++ -target %s %%*\r\n' "$ZIG" "$TARGET" > "$WRAP/zcxx.cmd"
printf '@"%s" ar %%*\r\n' "$ZIG" > "$WRAP/zar.cmd"
printf '@"%s" ranlib %%*\r\n' "$ZIG" > "$WRAP/zranlib.cmd"

BUILD="$HERE/build/x$BITS"
"$TOOLS/cmake.exe" -S "$HERE" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$(cygpath -m "$TOOLS/ninja.exe")" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(cygpath -m "$WRAP/zcc.cmd")" \
  -DCMAKE_CXX_COMPILER="$(cygpath -m "$WRAP/zcxx.cmd")" \
  -DCMAKE_AR="$(cygpath -m "$WRAP/zar.cmd")" \
  -DCMAKE_RANLIB="$(cygpath -m "$WRAP/zranlib.cmd")"
"$TOOLS/cmake.exe" --build "$BUILD"
"$BUILD/tflm_check.exe" "$HERE/vectors"
