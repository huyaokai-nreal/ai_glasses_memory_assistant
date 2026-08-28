# Android Local Demo

This module is an Android shell around the repository's existing Python core and web UI. It must not contain copied planner, memory, SQLite, privacy, or audit implementations.

## Toolchain

- JDK 17
- Android SDK 35 with build-tools 35.x
- An arm64 Android 8+ device. The current APK does not package an x86_64 Chaquopy runtime.

Set `ANDROID_HOME` and `JAVA_HOME`, then run:

```bash
cd android
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

To build, install, and launch the Debug APK on one USB/Wi-Fi ADB device in one command, run this from the repository root:

```bash
android/tools/build_and_install_debug.sh
```

The script finds common macOS JDK 17 and Android SDK locations, refuses to guess when multiple devices are connected, and uses `adb install -r` so a debug update normally preserves the app's data. Select a specific device when needed:

```bash
android/tools/build_and_install_debug.sh --serial DEVICE_SERIAL
```

The build reads `../ai_glasses_memory_assistant/**/*.py` directly through a filtered Chaquopy source set and adds only the Android-compatible Python dependency `numpy`. The four files in `../static/` are consumed as Android assets and extracted to app-private storage at runtime. The official sherpa-onnx 1.13.4 AAR is downloaded into `app/build/verified-dependencies` and accepted only when its pinned SHA-256 matches.

On first launch, enter the tester's DeepSeek provider, model, HTTPS base URL, and API key. The key is encrypted with Android Keystore and is not written to the Python SQLite databases. A complete `model_pack.v1` HTTPS manifest can be entered on the same settings screen; model files are downloaded by a foreground data-sync service, checked by size and SHA-256, and atomically activated in app-private storage. Stop continuous capture before updating models. After installation, run the five-component model self-test in Settings before starting capture.

On Android and mobile browsers, the chat header keeps only the app title and a settings gear. The in-app settings center owns speaker enrollment, TTS, location, memory, debug, and Android's advanced model/service settings entry. Memory management is a full-screen secondary page with an explicit back button. Android system Back closes speaker enrollment, debug, memory, or settings before navigating the WebView or leaving the app. The native advanced settings page remains responsible only for provider/API configuration, model installation and self-test, encrypted diagnostic export, a one-shot input test, and a Settings-only offline audio model test. The latter accepts PCM16 WAV (one or two channels) or AAC M4A such as a Lark download. M4A is decoded with the Android system decoder, then both formats are converted to 16 kHz mono, may receive a user-enabled -24 dB to +24 dB gain, and run the installed native Silero VAD and SenseVoice ASR per detected segment. ALAC, DRM-protected M4A, files without an audio track, or unavailable system decoders are rejected with a visible reason. The tool only displays the current result and clears the selected file, samples, and transcript when Settings stops; it never creates an audio event or writes to Python, Timeline, SQLite, audit, memory jobs, or diagnostics. It tests the Android model pipeline but not microphone routing, acoustic pickup, wake, speaker, or chat behavior. Every native microphone entry uses the same 16 kHz mono `VOICE_RECOGNITION` and input-routing contract: no Bluetooth microphone permits Android's system input; exactly one Bluetooth microphone is set as the preferred device and must match the actual `AudioRecord` route; multiple Bluetooth microphones, a preference failure, or a route mismatch stops capture instead of falling back to the phone microphone. Device changes recreate the recorder and re-evaluate this policy. The settings test displays the actual route, peak level, and local ASR text when its model is installed. It keeps at most five seconds of probe PCM only until the user plays it once, starts another test, or leaves Settings; then it clears the buffer. Probe PCM and its transcript are never sent to Python, added to Timeline/SQLite/audit, or included in diagnostics. Playback confirms that this short in-memory sample is audible; actual input-route verification remains the proof of the microphone source. This does not replace wake, ASR, memory, reply, or TTS acceptance.

For connected-device development, a manifest may instead declare `"install_mode": "adb_local"` and omit every file URL. Such a pack is rejected by the network downloader and can only be installed with the verified local tool:

```bash
python3 tools/install_local_model_pack.py \
  --serial DEVICE_SERIAL \
  --pack-dir /absolute/path/to/model-pack
```

The tool validates declared paths, sizes, and SHA-256 values on the Mac, verifies hashes again in the app-private directory, and only then atomically switches `files/models/current.json`.

Run the privacy-preserving connected-device smoke test with an explicit serial. It reports only public counters, model states, audit record types, memory/temperature metrics, and coarse location status; it never exports keys, databases, transcript text, PCM, embeddings, or coordinates:

```bash
python3 tools/run_device_acceptance.py \
  --serial DEVICE_SERIAL \
  --speech \
  --location \
  --stop-after
```

Reports are written under the ignored `android/captures/` directory. Any permission changed by the tool is restored in its `finally` path, and every temporary ADB forward is removed.
If the Mac speaker is too far from the device for VAD to trigger, repeat the acoustic transport check with `--speech-source device`; the report records that fallback explicitly, and it does not count as real-distance microphone accuracy.

### Mic Pro Bluetooth and USB acceptance

Being paired or connected to the Insta360 App over BLE does not mean Android exposes the Mic Pro as an audio input. The assistant only accepts an actual `AudioRecord` route; it will not treat a paired BLE control connection as a microphone or silently claim a phone-microphone fallback is Mic Pro audio.

Run each controlled scenario with a new report. Before each run, use the Android Settings input test to say a distinct phrase and play the five-second in-memory recording once; neither PCM nor its transcript is exported by the test tool.

```bash
# 1. Mic Pro connected as an Android Bluetooth input, Insta360 App not controlling it.
python3 android/tools/run_device_acceptance.py --serial DEVICE_SERIAL \
  --scenario mic-pro-system-bluetooth --expected-input bluetooth --stop-after

# 2. Insta360 App remains connected to Mic Pro over BLE while the assistant records.
python3 android/tools/run_device_acceptance.py --serial DEVICE_SERIAL \
  --scenario mic-pro-insta360-ble --expected-input bluetooth --stop-after

# 3. Official Mic Pro receiver connected as USB audio while Insta360 App retains BLE control.
python3 android/tools/run_device_acceptance.py --serial DEVICE_SERIAL \
  --scenario mic-pro-usb-receiver --expected-input usb --stop-after
```

`actual_input_route` must pass in every report. A Bluetooth result requires source `bluetooth`; the USB fallback requires `usb`. If the BLE scenario reports no Bluetooth route or a route other than `bluetooth`, the S22, current Mic Pro firmware, and current Insta360 App session are not concurrently compatible. Do not try to force-close or seize the other App's connection from this application; use the USB receiver topology and keep the App's BLE link for device management.

Pull a complete, local-only usage snapshot from a USB-connected debug APK. Use it after you have used the app for a while and want Codex to investigate response feedback, ASR, memory storage or recall quality:

```bash
conda run -n hermes python android/tools/pull_device_usage_snapshot.py \
  --serial DEVICE_SERIAL
```

The tool first stops continuous capture, then waits for the audio queue and the final memory/archive job. If that does not finish in time, it refuses to create a partial snapshot. On success it creates a temporary debug-only ZIP in app-private cache, streams it to `android/captures/usage-data/`, validates every manifest checksum, extracts it with current-user-only permissions, and removes only that temporary phone ZIP. It never deletes the app database, existing memories, or model packs; capture stays stopped after the export.

Each timestamped directory contains `database/timeline.db` (original chat, replies, final ASR, feedback, discussion/day summaries), `database/events.db` (structured memory), `database/sessions.db` when present, `audit/chat_audit.jsonl`, `manifest.json`, `feedback_index.json`, and `analysis_request.md`. The content stays on the Mac and is not sent to any service or agent automatically. It preserves original product text, including your feedback notes, but excludes API-key/authorization values, raw PCM or encoded raw audio, and voice-profile/enrollment embeddings. Send the printed directory path to Codex; an agent should start at `feedback_index.json`, prioritize `needs_improvement`, and use its `turn_id` to trace Timeline and audit evidence. When one ready device is connected, `--serial` may be omitted; when multiple devices are present the tool lists valid choices. This ADB path is intentionally limited to debuggable APKs and does not work with release builds.

`model-pack.example.json` documents the required roles. It is a schema example only: its placeholder URLs, sizes, and hashes are intentionally not installable. Every path referenced by a component must also appear in `files` with the exact production byte size and lowercase SHA-256.

The current APK contains the foreground microphone service, durable external-event ingestion, streaming partial UI, conservative overlap evidence, system TTS, and compiled sherpa adapters for VAD, KWS, online ASR, SenseVoice, and speaker embeddings. Native speaker enrollment records three complete VAD segments and sends private embeddings to the shared Python aggregation path; Kotlin never writes the profile database directly.

Continuous capture now distinguishes ambient listening from wake recognition in both the compact chat status and the native bridge. The user may say `你好小忆，问题` in one utterance, or say `你好小忆`, wait for `我在，请说`, and then ask within 10 seconds. Only the final question without the wake phrase is displayed as a user message and dispatched; partial transcription remains UI-only. A lightweight `audioUiStatus()` bridge carries wake progress while the existing `audioStatus()` remains the slower diagnostics/status call.

`x4000-sherpa-1.13.4-v1` has passed all five instrumentation self-tests on an XREAL X4000 (Android 14). A short locked-screen capture also passed with 16 kHz mono PCM16 remaining unsilenced and about 690 MB total PSS while all models were resident. These checks do not replace human acoustic acceptance: three-sample owner enrollment, real `你好小忆` wake/query/TTS, echo, second-speaker privacy, failure recovery, and 8/24-hour endurance are still required before 24-hour delivery can be claimed.

Settings can export a password-encrypted `.aigd` diagnostic bundle while capture is stopped. The bundle contains sanitized SQLite snapshots, redacted audit records, model/runtime state, battery and memory metrics, and app/device versions. It explicitly excludes API keys, raw PCM, speaker profiles, enrollment samples, and private embedding payloads. Decrypt it on the development Mac without putting the password in shell history:

```bash
java android/tools/DecryptDiagnosticBundle.java /path/to/report.aigd
```

The tool prompts for the password, writes a sibling `.zip`, and refuses to overwrite an existing output.
