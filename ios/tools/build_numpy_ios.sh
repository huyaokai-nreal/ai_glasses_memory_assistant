#!/bin/sh
set -eu

# Cross-compile numpy 1.26.2 for iOS arm64.
# Prerequisites: Xcode with iOS SDK, the Python.xcframework at ios/.deps/.
#
# Usage:
#   ./ios/tools/build_numpy_ios.sh
#
# Output:
#   ios/.deps/numpy-arm64/numpy/  (ready to be copied into PythonRuntime)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_DIR/ios/.deps"
PYTHON_XCF="$DEPS_DIR/Python.xcframework/ios-arm64"
NUMPY_VERSION="1.26.2"
NUMPY_BUILD_DIR="$DEPS_DIR/numpy-build"
NUMPY_OUTPUT_DIR="$DEPS_DIR/numpy-arm64"

# Locate Xcode and SDK.
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
SDK_PATH="$DEVELOPER_DIR/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk"
if [ ! -d "$SDK_PATH" ]; then
  echo "ERROR: iPhoneOS SDK not found at $SDK_PATH" >&2
  echo "Install Xcode or set DEVELOPER_DIR to a valid Xcode path." >&2
  exit 2
fi

# Python framework paths.
PYTHON_FRAMEWORK="$PYTHON_XCF/Python.framework"
PYTHON_HEADERS="$PYTHON_XCF/include/python3.11"
PYTHON_LIB_DIR="$PYTHON_XCF/lib-arm64/python3.11"

if [ ! -d "$PYTHON_FRAMEWORK" ] || [ ! -f "$PYTHON_HEADERS/Python.h" ]; then
  echo "ERROR: Python.xcframework not found at $PYTHON_XCF" >&2
  echo "Place the CPython 3.11.15 xcframework at ios/.deps/Python.xcframework first." >&2
  exit 2
fi

# Compiler name prefix (matches Beeware/Python-Apple-support convention).
# Xcode's clang must be invoked with the right target triple for iOS arm64.
CC="xcrun --sdk iphoneos clang -target arm64-apple-ios13.0"
CXX="xcrun --sdk iphoneos clang++ -target arm64-apple-ios13.0"

echo "==> Building numpy $NUMPY_VERSION for iOS arm64"

# Download numpy source if needed.
NUMPY_SRC="$NUMPY_BUILD_DIR/numpy-$NUMPY_VERSION"
if [ ! -d "$NUMPY_SRC" ]; then
  mkdir -p "$NUMPY_BUILD_DIR"
  if [ ! -f "$NUMPY_BUILD_DIR/numpy-$NUMPY_VERSION.tar.gz" ]; then
    echo "==> Downloading numpy $NUMPY_VERSION..."
    curl -fsSL "https://github.com/numpy/numpy/releases/download/v$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz" \
      -o "$NUMPY_BUILD_DIR/numpy-$NUMPY_VERSION.tar.gz"
  fi
  echo "==> Extracting numpy $NUMPY_VERSION..."
  tar xzf "$NUMPY_BUILD_DIR/numpy-$NUMPY_VERSION.tar.gz" -C "$NUMPY_BUILD_DIR"
fi

cd "$NUMPY_SRC"

# Write site.cfg for iOS cross-compilation.
# Use Accelerate.framework for BLAS/LAPACK on iOS.
cat > site.cfg <<'EOF'
[DEFAULT]
library_dirs =
include_dirs =

[accelerate]
libraries = Accelerate
library_dirs =
include_dirs =

[blas]
libraries = Accelerate

[blas_opt]
libraries = Accelerate

[lapack]
libraries = Accelerate

[lapack_opt]
libraries = Accelerate
EOF

# Point to host Python for build-time code generation.
HOST_PYTHON="${HOST_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [ ! -x "$HOST_PYTHON" ]; then
  HOST_PYTHON="$(command -v python3.11 || command -v python3)"
fi
if [ ! -x "$HOST_PYTHON" ]; then
  echo "ERROR: host python3 not found; set HOST_PYTHON env var" >&2
  exit 2
fi
if ! "$HOST_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
  echo "ERROR: NumPy 1.26.2 iOS build requires host Python 3.11; set HOST_PYTHON explicitly" >&2
  exit 2
fi
if ! "$HOST_PYTHON" -c 'import Cython' >/dev/null 2>&1; then
  echo "ERROR: host Python is missing Cython; install Cython<3 in the build environment" >&2
  exit 2
fi

# Build flags matching the CPython cross-compile. PYTHONHOME is intentionally
# not set: it would make the host interpreter search the iPhoneOS bundle for
# its own stdlib before NumPy's build hooks can run.
TARGET_CONFIG_DIR="$PYTHON_XCF/platform-config/arm64-iphoneos"
TARGET_STDLIB_DIR="$PYTHON_XCF/lib-arm64/python3.11"
export _PYTHON_HOST_PLATFORM="ios-arm64-iphoneos"
export _PYTHON_SYSCONFIGDATA_NAME="_sysconfigdata__ios_arm64-iphoneos"
export PYTHONPATH="$TARGET_CONFIG_DIR:$PYTHON_LIB_DIR:$TARGET_STDLIB_DIR"
# NumPy 1.26's distutils glue imports the legacy msvccompiler module even on
# Darwin.  The stdlib copy still contains that compatibility module; recent
# setuptools' vendored copy does not.
export SETUPTOOLS_USE_DISTUTILS="stdlib"
export CC="$CC"
export CXX="$CXX"
export AR="$(xcrun --sdk iphoneos -f ar)"
export RANLIB="$(xcrun --sdk iphoneos -f ranlib)"
export CPPFLAGS="-I$PYTHON_HEADERS -I$PYTHON_XCF/Python.framework/Headers"
export CFLAGS="$CPPFLAGS -isysroot $SDK_PATH -mios-version-min=13.0 -arch arm64 -O3"
export CXXFLAGS="$CPPFLAGS -isysroot $SDK_PATH -mios-version-min=13.0 -arch arm64 -O3"
export LDFLAGS="-isysroot $SDK_PATH -mios-version-min=13.0 -arch arm64 -F$PYTHON_XCF -framework Python"
export LDSHARED="$CC -dynamiclib -mios-version-min=13.0 -F$PYTHON_XCF -framework Python"
export BLDSHARED="$CC -dynamiclib -mios-version-min=13.0 -F$PYTHON_XCF -framework Python"

# numpy-specific: disable features that won't work on iOS.
export NPY_DISABLE_SVML=1
export NPY_BLAS_ORDER="accelerate"
export NPY_LAPACK_ORDER="accelerate"

echo "==> Building numpy (this may take several minutes)..."
"$HOST_PYTHON" setup.py build --force build_ext --inplace

echo "==> Installing numpy to $NUMPY_OUTPUT_DIR..."
rm -rf "$NUMPY_OUTPUT_DIR"
mkdir -p "$NUMPY_OUTPUT_DIR"

# Copy the built numpy package to the output directory.
# The build creates numpy/*.so and numpy/ directories in the source tree.
if [ -d "numpy" ]; then
  # Copy pure Python files and platform-specific .so files.
  "$HOST_PYTHON" setup.py install \
    --root="$NUMPY_OUTPUT_DIR" \
    --prefix=""

  # Flatten: move from lib/python3.11/site-packages/numpy to numpy-arm64/numpy/
  NUMPY_INSTALLED=$(find "$NUMPY_OUTPUT_DIR" -name "numpy" -type d -path "*/site-packages/*" 2>/dev/null | head -1)
  if [ -n "$NUMPY_INSTALLED" ]; then
    mv "$NUMPY_INSTALLED" "$NUMPY_OUTPUT_DIR/numpy"
    find "$NUMPY_OUTPUT_DIR" -not -path "$NUMPY_OUTPUT_DIR/numpy*" -delete
  else
    # If install didn't work, try to copy the in-place build.
    cp -R numpy "$NUMPY_OUTPUT_DIR/numpy"
  fi
else
  echo "ERROR: numpy build directory not found" >&2
  exit 1
fi

echo "==> Verifying numpy build..."
# Check for .so files (native extensions).
SO_COUNT=$(find "$NUMPY_OUTPUT_DIR/numpy" -type f \( -name "*.so" -o -name "*.cpython-311-iphoneos.so" \) | wc -l | tr -d ' ')
if [ "$SO_COUNT" -eq 0 ]; then
  echo "ERROR: numpy build produced no native extensions" >&2
  exit 1
fi
echo "  Built $SO_COUNT native extension(s)"

# Verify arm64 slice.
while IFS= read -r so; do
  if ! lipo -info "$so" 2>&1 | grep -q "arm64"; then
    echo "ERROR: non-arm64 native extension: $so" >&2
    exit 1
  fi
  echo "  arm64: $(basename "$so")"
done <<EOF
$(find "$NUMPY_OUTPUT_DIR/numpy" -type f \( -name "*.so" -o -name "*.cpython-311-iphoneos.so" \))
EOF

if ! find "$NUMPY_OUTPUT_DIR/numpy/core" -type f -name '_multiarray_umath*.so' | grep -q .; then
  echo "ERROR: numpy _multiarray_umath extension is missing" >&2
  exit 1
fi

echo "==> numpy $NUMPY_VERSION build complete"
echo "Output: $NUMPY_OUTPUT_DIR/numpy/"
echo "Xcode resource copy phase will fail closed if this output is missing."
