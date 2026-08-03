#!/bin/sh
set -eu

# Build and install AIGlassesMemoryAssistant to a connected iPhone.
# Usage: ./ios/tools/build_and_install.sh --serial SERIAL

SERIAL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --serial) SERIAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$SERIAL" ]; then
    echo "ERROR: --serial SERIAL is required; verify the device with devicectl first." >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "$0")/../AIGlassesMicProbe" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
DERIVED_DATA_PATH="${AI_GLASSES_IOS_DERIVED_DATA:-${TMPDIR:-/tmp}/ai-glasses-memory-assistant-deriveddata}"

echo "==> Building AIGlassesMemoryAssistant for device..."
cd "$PROJECT_DIR"
DEVELOPER_DIR="$DEVELOPER_DIR" xcodebuild \
    -project AIGlassesMicProbe.xcodeproj \
    -scheme AIGlassesMemoryAssistant \
    -sdk iphoneos \
    -configuration Debug \
    -destination "id=$SERIAL" \
    -derivedDataPath "$DERIVED_DATA_PATH" \
    -allowProvisioningUpdates \
    build

APP_PATH="$DERIVED_DATA_PATH/Build/Products/Debug-iphoneos/AIGlassesMemoryAssistant.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Build product not found at $APP_PATH"
    exit 1
fi

echo "==> Build OK: $APP_PATH"
python3 "$REPO_DIR/ios/tools/verify_ios_dependencies.py" \
    --deps-dir "$REPO_DIR/ios/.deps" \
    --app-path "$APP_PATH"

# Install to device
echo "==> Installing to device..."
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun devicectl device install app --device "$SERIAL" "$APP_PATH"

echo "==> Done. App bundle is ready at:"
echo "    $APP_PATH"
