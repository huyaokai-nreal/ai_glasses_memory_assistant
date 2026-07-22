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

The build reads `../ai_glasses_memory_assistant/**/*.py` directly through a filtered Chaquopy source set and adds only the Android-compatible Python dependency `numpy`. The four files in `../static/` are consumed as Android assets and extracted to app-private storage at runtime. The official sherpa-onnx 1.13.4 AAR is downloaded into `app/build/verified-dependencies` and accepted only when its pinned SHA-256 matches.

On first launch, enter the tester's DeepSeek provider, model, HTTPS base URL, and API key. The key is encrypted with Android Keystore and is not written to the Python SQLite databases. A complete `model_pack.v1` HTTPS manifest can be entered on the same settings screen; model files are downloaded by a foreground data-sync service, checked by size and SHA-256, and atomically activated in app-private storage. Stop continuous capture before updating models. After installation, run the five-component model self-test in Settings before starting capture.

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

`model-pack.example.json` documents the required roles. It is a schema example only: its placeholder URLs, sizes, and hashes are intentionally not installable. Every path referenced by a component must also appear in `files` with the exact production byte size and lowercase SHA-256.

The current APK contains the foreground microphone service, durable external-event ingestion, streaming partial UI, conservative overlap evidence, system TTS, and compiled sherpa adapters for VAD, KWS, online ASR, SenseVoice, and speaker embeddings. Native speaker enrollment records three complete VAD segments and sends private embeddings to the shared Python aggregation path; Kotlin never writes the profile database directly.

`x4000-sherpa-1.13.4-v1` has passed all five instrumentation self-tests on an XREAL X4000 (Android 14). A short locked-screen capture also passed with 16 kHz mono PCM16 remaining unsilenced and about 690 MB total PSS while all models were resident. These checks do not replace human acoustic acceptance: three-sample owner enrollment, real `你好小忆` wake/query/TTS, echo, second-speaker privacy, failure recovery, and 8/24-hour endurance are still required before 24-hour delivery can be claimed.

Settings can export a password-encrypted `.aigd` diagnostic bundle while capture is stopped. The bundle contains sanitized SQLite snapshots, redacted audit records, model/runtime state, battery and memory metrics, and app/device versions. It explicitly excludes API keys, raw PCM, speaker profiles, enrollment samples, and private embedding payloads. Decrypt it on the development Mac without putting the password in shell history:

```bash
java android/tools/DecryptDiagnosticBundle.java /path/to/report.aigd
```

The tool prompts for the password, writes a sibling `.zip`, and refuses to overwrite an existing output.
