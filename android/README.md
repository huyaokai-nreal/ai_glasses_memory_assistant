# Android Local Demo

This module is an Android shell around the repository's existing Python core and web UI. It must not contain copied planner, memory, SQLite, privacy, or audit implementations.

## Toolchain

- JDK 17
- Android SDK 35 with build-tools 35.x
- An arm64 Android 8+ device, or an x86_64 emulator for debug

Set `ANDROID_HOME` and `JAVA_HOME`, then run:

```bash
cd android
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

The build reads `../ai_glasses_memory_assistant/**/*.py` directly through a filtered Chaquopy source set and adds only the Android-compatible Python dependency `numpy`. The four files in `../static/` are consumed as Android assets and extracted to app-private storage at runtime. The official sherpa-onnx 1.13.4 AAR is downloaded into `app/build/verified-dependencies` and accepted only when its pinned SHA-256 matches.

On first launch, enter the tester's DeepSeek provider, model, HTTPS base URL, and API key. The key is encrypted with Android Keystore and is not written to the Python SQLite databases. A complete `model_pack.v1` HTTPS manifest can be entered on the same settings screen; model files are downloaded by a foreground data-sync service, checked by size and SHA-256, and atomically activated in app-private storage. Stop continuous capture before updating models.

`model-pack.example.json` documents the required roles. It is a schema example only: its placeholder URLs, sizes, and hashes are intentionally not installable. Every path referenced by a component must also appear in `files` with the exact production byte size and lowercase SHA-256.

The current APK contains the foreground microphone service, durable external-event ingestion, and compiled sherpa adapters for VAD, KWS, online ASR, SenseVoice, and speaker embeddings. Android speaker enrollment, a production model manifest, and physical-device endurance remain separate acceptance gates and must not be claimed until they pass.
