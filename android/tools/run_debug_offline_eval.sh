#!/usr/bin/env bash
# Debug-only: drive one Eval_Ali offline replay via OfflineEvalDebugReceiver.
# Reads the WAV from --wav (host path), pushes it into the app's private
# filesDir/eval_ali_input/, fires the debug broadcast, then pulls the JSON
# from filesDir/eval_ali_exports/ into --out (host path).
#
# Usage:
#   bash android/tools/run_debug_offline_eval.sh \
#       --serial <SERIAL> \
#       --wav <host.wav> \
#       --case-id <case_id> \
#       --out <host.json>
#
# Exits 0 on success, non-zero on broadcast failure or timeout.
set -euo pipefail

SERIAL=""
WAV=""
CASE_ID=""
OUT=""
TIMEOUT_SECONDS=180

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial) SERIAL="$2"; shift 2 ;;
        --wav) WAV="$2"; shift 2 ;;
        --case-id) CASE_ID="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$SERIAL" || -z "$WAV" || -z "$CASE_ID" || -z "$OUT" ]]; then
    echo "Usage: $0 --serial SERIAL --wav PATH --case-id ID --out PATH [--timeout SECONDS]" >&2
    exit 2
fi

PKG="com.aiglasses.memoryassistant.demo"
ADB="adb"
WAV_NAME="$(basename "$WAV")"
TMP_REMOTE="/data/local/tmp/$WAV_NAME"

# 1. push WAV to a world-readable staging area on device
$ADB -s "$SERIAL" push "$WAV" "$TMP_REMOTE" > /dev/null 2>&1

# 2. copy into the app's private input dir via separate run-as calls
#    (avoids fragile sh -c quoting through adb shell).
$ADB -s "$SERIAL" shell run-as "$PKG" mkdir -p files/eval_ali_input
$ADB -s "$SERIAL" shell run-as "$PKG" mkdir -p files/eval_ali_exports
$ADB -s "$SERIAL" shell run-as "$PKG" cp "$TMP_REMOTE" "files/eval_ali_input/$WAV_NAME"
$ADB -s "$SERIAL" shell rm -f "$TMP_REMOTE"

# 3. fire the debug broadcast and wait up to TIMEOUT_SECONDS for completion
RESULT=$($ADB -s "$SERIAL" shell am broadcast \
    -a "com.aiglasses.memoryassistant.debug.RUN_OFFLINE_EVAL" \
    -p "$PKG" \
    --es wav_name "$WAV_NAME" \
    --es case_id "$CASE_ID" \
    -W 2>&1) || true

if ! grep -q 'result=0' <<<"$RESULT"; then
    echo "broadcast did not succeed: $RESULT" >&2
    $ADB -s "$SERIAL" shell rm -f "$TMP_REMOTE" > /dev/null 2>&1 || true
    exit 1
fi

# 4. poll for the JSON (the receiver writes after the pipeline returns)
DEADLINE=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < DEADLINE )); do
    if $ADB -s "$SERIAL" shell run-as "$PKG" \
        test -f "files/eval_ali_exports/$CASE_ID.json" 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! $ADB -s "$SERIAL" shell run-as "$PKG" \
    test -f "files/eval_ali_exports/$CASE_ID.json" 2>/dev/null; then
    echo "timed out waiting for $CASE_ID.json after ${TIMEOUT_SECONDS}s" >&2
    exit 1
fi

# 5. pull the JSON to host
$ADB -s "$SERIAL" exec-out run-as "$PKG" cat "files/eval_ali_exports/$CASE_ID.json" > "$OUT"
echo "ok $CASE_ID -> $OUT"