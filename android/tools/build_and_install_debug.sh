#!/usr/bin/env bash
# Build the Debug APK, install it on one explicitly selected USB/Wi-Fi device, and launch it.
#
# This script installs ONLY the APK. It uses `adb install -r`, which keeps the
# app's private data (including the speech model pack in files/models/), so the
# model you installed once with install_local_model_pack.sh is preserved. This
# script never clears app data and never installs the model.
set -euo pipefail

PACKAGE_NAME="com.aiglasses.memoryassistant.demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APK_PATH="$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
SERIAL=""

usage() {
    cat <<'EOF'
Usage: android/tools/build_and_install_debug.sh [--serial DEVICE_SERIAL]

Builds the Debug APK, installs it with adb install -r, and starts the app.
When more than one ready device is connected, --serial is required.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial)
            [[ $# -ge 2 ]] || { echo "--serial requires a device serial" >&2; exit 2; }
            SERIAL="$2"
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

if [[ -z "${JAVA_HOME:-}" ]]; then
    for candidate in \
        /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
        /usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home; do
        if [[ -x "$candidate/bin/java" ]]; then
            export JAVA_HOME="$candidate"
            break
        fi
    done
fi

[[ -x "${JAVA_HOME:-}/bin/java" ]] || {
    echo "JDK 17 is required. Set JAVA_HOME or install openjdk@17." >&2
    exit 1
}
JAVA_VERSION="$($JAVA_HOME/bin/java -version 2>&1 | head -n 1)"
[[ "$JAVA_VERSION" == *'"17.'* || "$JAVA_VERSION" == *'"17"'* ]] || {
    echo "JDK 17 is required, found: $JAVA_VERSION" >&2
    exit 1
}

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
if [[ -z "${ANDROID_HOME:-}" && "$(basename "$(dirname "$ADB")")" == "platform-tools" ]]; then
    SDK_ROOT="$(cd "$(dirname "$ADB")/.." && pwd)"
    if [[ -d "$SDK_ROOT/platforms" && -d "$SDK_ROOT/build-tools" ]]; then
        export ANDROID_HOME="$SDK_ROOT"
    fi
fi
[[ -d "${ANDROID_HOME:-}/platforms" && -d "${ANDROID_HOME:-}/build-tools" ]] || {
    echo "Android SDK platforms/build-tools were not found. Set ANDROID_HOME to a complete Android SDK." >&2
    exit 1
}

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

echo "Building Debug APK..."
(cd "$ANDROID_DIR" && ./gradlew :app:assembleDebug)
[[ -f "$APK_PATH" ]] || { echo "Debug APK was not produced: $APK_PATH" >&2; exit 1; }

echo "Installing on $SERIAL..."
"$ADB" -s "$SERIAL" install -r "$APK_PATH"
echo "Launching $PACKAGE_NAME..."
"$ADB" -s "$SERIAL" shell monkey -p "$PACKAGE_NAME" -c android.intent.category.LAUNCHER 1 >/dev/null
echo "Installed and launched: $APK_PATH"

# Non-fatal check: the speech model pack lives in app-private storage and is
# installed separately via install_local_model_pack.sh. Confirm it is present so
# the user knows if they skipped the one-time model install.
if "$ADB" -s "$SERIAL" shell run-as "$PACKAGE_NAME" test -f files/models/current.json >/dev/null 2>&1; then
    echo "Speech model pack: present (preserved across APK reinstalls)."
else
    echo "WARN: speech model pack not found on $SERIAL."
    echo "      Install it once with: android/tools/install_local_model_pack.sh --serial $SERIAL"
    echo "      (this script only installs the APK and leaves the model untouched)"
fi
