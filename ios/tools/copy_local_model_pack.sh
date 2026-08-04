pack="${SRCROOT}/../.local-models/x4000-sherpa-1.13.4-v2"
destination="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Models"
rm -rf "$destination"
mkdir -p "$destination"
ditto "$pack" "$destination"
web_source="${SRCROOT}/../../static"
web_destination="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Web"
rm -rf "$web_destination"
mkdir -p "$web_destination/static"
ditto "$web_source" "$web_destination/static"
sed 's|"/static/|"static/|g' "$web_source/index.html" > "$web_destination/index.html"
python_destination="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/PythonRuntime"
rm -rf "$python_destination"
mkdir -p "$python_destination/lib/python3.11" "$python_destination/python"
# Step 1: Copy platform-specific files from xcframework (lib-dynload .so, sysconfig).
python_support="${SRCROOT}/../.deps/Python.xcframework/ios-arm64"
if [ -d "$python_support/lib-arm64/python3.11" ]; then
  ditto "$python_support/lib-arm64/python3.11" "$python_destination/lib/python3.11"
fi
# Step 2: Copy pure-Python stdlib from xcframework root-level lib/.
stdlib="${SRCROOT}/../.deps/Python.xcframework/lib/python3.11"
if [ -d "$stdlib" ]; then
  echo "Python stdlib source: $stdlib"
  cp -R "$stdlib"/* "$python_destination/lib/python3.11/"
  rm -rf "$python_destination/lib/python3.11/__pycache__"
  rm -rf "$python_destination/lib/python3.11/test"
  rm -rf "$python_destination/lib/python3.11/idlelib"
  rm -rf "$python_destination/lib/python3.11/turtledemo"
  rm -rf "$python_destination/lib/python3.11/tkinter"
  rm -rf "$python_destination/lib/python3.11/ensurepip"
  rm -rf "$python_destination/lib/python3.11/distutils"
  rm -rf "$python_destination/lib/python3.11/venv"
  rm -rf "$python_destination/lib/python3.11/site-packages"
  require() { if [ ! -e "$1" ]; then echo "ERROR: $2" >&2; exit 1; fi; }
  require "$python_destination/lib/python3.11/encodings/__init__.py" "encodings missing"
  require "$python_destination/lib/python3.11/_apple_support.py" "_apple_support missing"
  echo "Python stdlib merged (platform .so kept from lib-arm64)"
else
  echo "ERROR: xcframework lib/ not found at $stdlib" >&2
  exit 1
fi
if [ -d "${SRCROOT}/../../ai_glasses_memory_assistant" ]; then
  ditto "${SRCROOT}/../../ai_glasses_memory_assistant" "$python_destination/python/ai_glasses_memory_assistant"
fi
numpy_src="${SRCROOT}/../.deps/numpy-arm64/numpy"
if [ ! -d "$numpy_src" ]; then
  echo "ERROR: bundled numpy is missing at $numpy_src; run ios/tools/build_numpy_ios.sh first" >&2
  exit 1
fi
numpy_dest="$python_destination/lib/python3.11/site-packages/numpy"
mkdir -p "$(dirname "$numpy_dest")"
ditto "$numpy_src" "$numpy_dest"
echo "numpy $numpy_src -> $numpy_dest"

# Step 3: Code-sign all .so files for iOS sandbox.
# lib-dynload contains CPython built-ins; site-packages/* contains third-party
# extensions (numpy, _sqlite3, _ssl, etc.). Both must be signed.
identity="${CODE_SIGN_IDENTITY:--}"
echo "Code signing Python .so modules (identity: $identity)..."
find "$python_destination/lib/python3.11" -name "*.so" -type f | while read -r so; do
  codesign --force --sign "$identity" "$so" 2>/dev/null || {
    echo "WARNING: failed to sign $(basename "$so"), trying ad-hoc..."
    codesign --force --sign - "$so" 2>/dev/null || echo "WARNING: ad-hoc failed for $(basename "$so")"
  }
done
echo "Python .so modules signed"
