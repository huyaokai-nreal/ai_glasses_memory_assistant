#!/usr/bin/env bash
# Install a local (Mac-side) adb_local model pack onto one Android device, once.
#
# The speech model pack is large and lives in the app's private storage
# (files/models/...). It is installed ONCE with this script and then preserved
# across APK reinstalls done by build_and_install_debug.sh (that script uses
# `adb install -r`, which keeps app data). Do NOT reinstall the model on every
# code change; only rerun this script when you actually swap the model pack.
set -euo pipefail

PACKAGE_NAME="com.aiglasses.memoryassistant.demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Default to the only real adb_local pack shipped in this repo.
DEFAULT_PACK_DIR="$REPO_DIR/ios/.local-models/x4000-sherpa-1.13.4-ten-sensevoice-2025-v2"
SERIAL=""
PACK_DIR="$DEFAULT_PACK_DIR"

usage() {
    cat <<'EOF'
Usage: android/tools/install_local_model_pack.sh [--serial DEVICE_SERIAL] [--pack-dir DIR]

Installs a local model pack (adb_local) from this Mac onto one Android device.
Run this ONCE per device; later APK-only reinstalls keep the model in place.

  --serial DEVICE_SERIAL   Target device serial (required when >1 device is connected)
  --pack-dir DIR           Directory containing manifest.json (default: the repo's
                           ios/.local-models/x4000-sherpa-1.13.4-ten-sensevoice-2025-v2 pack)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial)
            [[ $# -ge 2 ]] || { echo "--serial requires a device serial" >&2; exit 2; }
            SERIAL="$2"
            shift 2
            ;;
        --pack-dir)
            [[ $# -ge 2 ]] || { echo "--pack-dir requires a path" >&2; exit 2; }
            PACK_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$PACK_DIR/manifest.json" ]]; then
    echo "Model pack not found at: $PACK_DIR" >&2
    echo "Expected a directory containing manifest.json. Use --pack-dir to point at it." >&2
    exit 1
fi

# --- adb discovery (same precedence as build_and_install_debug.sh) ---
if [[ -z "${ANDROID_HOME:-}" ]]; then
    for candidate in \
        "${ANDROID_SDK_ROOT:-}" \
        "$HOME/Library/Android/sdk" \
        /opt/homebrew/share/android-commandlinetools \
        /usr/local/share/android-commandlinetools; do
        if [[ -d "$candidate/platforms" && -d "$candidate/build-tools" ]]; then
            export ANDROID_HOME="$candidate"
            break
        fi
    done
fi

ADB="${ANDROID_HOME:-}/platform-tools/adb"
if [[ ! -x "$ADB" ]]; then
    ADB="$(command -v adb || true)"
fi
[[ -n "$ADB" && -x "$ADB" ]] || {
    echo "adb was not found. Set ANDROID_HOME or add platform-tools to PATH." >&2
    exit 1
}

# --- device selection (mirrors build_and_install_debug.sh) ---
if [[ -z "$SERIAL" ]]; then
    READY_DEVICES="$($ADB devices | awk 'NR > 1 && $2 == "device" { print $1 }')"
    READY_COUNT="$(printf '%s\n' "$READY_DEVICES" | sed '/^$/d' | wc -l | tr -d ' ')"
    case "$READY_COUNT" in
        0)
            echo "No ready Android device. Enable USB debugging, connect the device, and accept the computer prompt." >&2
            "$ADB" devices -l >&2
            exit 1
            ;;
        1)
            SERIAL="$READY_DEVICES"
            ;;
        *)
            echo "More than one ready device is connected. Choose one with --serial DEVICE_SERIAL:" >&2
            "$ADB" devices -l >&2
            exit 2
            ;;
    esac
fi

[[ "$($ADB -s "$SERIAL" get-state)" == "device" ]] || {
    echo "Device $SERIAL is not ready for adb." >&2
    exit 1
}

# --- python discovery: prefer the repo's conda env, fall back to python3 ---
PYTHON_BIN=""
if command -v conda >/dev/null 2>&1; then
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx hermes; then
        PYTHON_BIN="conda run -n hermes python"
    fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
    [[ -n "$PYTHON_BIN" ]] || { echo "python3 was not found." >&2; exit 1; }
fi

echo "Model pack : $PACK_DIR"
echo "Device     : $SERIAL"
echo ""
echo "NOTE: the model pack is large. Install it ONCE; later APK reinstalls via"
echo "      android/tools/build_and_install_debug.sh keep it in place (adb install -r)."
echo ""

# shellcheck disable=SC2086
$PYTHON_BIN "$SCRIPT_DIR/install_local_model_pack.py" --serial "$SERIAL" --pack-dir "$PACK_DIR"
echo "Model pack install finished on $SERIAL."
