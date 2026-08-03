#!/bin/sh
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
pack_dir="$script_dir/../.local-models/x4000-sherpa-1.13.4-v2"
destination="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/Models"
web_source="$script_dir/../../static"
web_destination="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/Web"
python_support="$script_dir/../.deps/Python.xcframework/ios-arm64"
python_destination="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/PythonRuntime"

python3 "$script_dir/verify_model_pack.py" --pack-dir "$pack_dir"
rm -rf "$destination"
mkdir -p "$destination"
ditto "$pack_dir" "$destination"
rm -rf "$web_destination"
mkdir -p "$web_destination/static"
ditto "$web_source" "$web_destination/static"
sed 's|"/static/|"static/|g' "$web_source/index.html" > "$web_destination/index.html"
rm -rf "$python_destination"
mkdir -p "$python_destination/lib" "$python_destination/python"
ditto "$python_support/lib-arm64/python3.11" "$python_destination/lib/python3.11"
ditto "$script_dir/../../ai_glasses_memory_assistant" "$python_destination/python/ai_glasses_memory_assistant"
numpy_src="$script_dir/../.deps/numpy-arm64/numpy"
if [ ! -d "$numpy_src" ]; then
  echo "ERROR: bundled numpy is missing at $numpy_src; run ios/tools/build_numpy_ios.sh first" >&2
  exit 1
fi
numpy_dest="$python_destination/lib/python3.11/site-packages/numpy"
mkdir -p "$(dirname "$numpy_dest")"
ditto "$numpy_src" "$numpy_dest"
echo "numpy $numpy_src -> $numpy_dest"
