#!/bin/sh
set -eu

# Build and install AIGlassesMemoryAssistant to a connected iPhone.
# Usage: ./ios/tools/build_and_install.sh [--serial SERIAL]
#   --serial SERIAL  Target a specific device by UDID (required if multiple devices connected).

SERIAL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --serial) SERIAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/../AIGlassesMicProbe" && pwd)"

echo "==> Building AIGlassesMemoryAssistant for device..."
cd "$PROJECT_DIR"
xcodebuild \
    -project AIGlassesMicProbe.xcodeproj \
    -scheme AIGlassesMemoryAssistant \
    -sdk iphoneos \
    -configuration Debug \
    -allowProvisioningUpdates \
    build 2>&1 | tail -5

APP_PATH="$HOME/Library/Developer/Xcode/DerivedData/AIGlassesMicProbe-"*/Build/Products/Debug-iphoneos/AIGlassesMemoryAssistant.app

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Build product not found at $APP_PATH"
    exit 1
fi

echo "==> Build OK: $APP_PATH"

# Install to device
DEVICE_FLAG=""
if [ -n "$SERIAL" ]; then
    DEVICE_FLAG="--id $SERIAL"
fi

echo "==> Installing to device..."
xcrun devicectl device install app --device "$SERIAL" "$APP_PATH" 2>/dev/null || \
    ios-deploy --bundle "$APP_PATH" --justlaunch 2>/dev/null || \
    echo "NOTE: Auto-install failed. Open in Xcode and select your device to install."

echo "==> Done. App bundle is ready at:"
echo "    $APP_PATH"
