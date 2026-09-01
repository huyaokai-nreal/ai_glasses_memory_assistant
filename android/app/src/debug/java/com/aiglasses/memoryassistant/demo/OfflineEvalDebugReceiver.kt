package com.aiglasses.memoryassistant.demo

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import java.io.File

/**
 * Debug-only entry point used by the Eval_Ali on-device replay harness.
 *
 * Runs [OfflineAudioTestRunner] on a WAV already placed in the app's own
 * private storage and writes the [OfflineAudioTestResult.toEvalAliDebugJson]
 * output next to it. Lives in `src/debug` so it is not included in release
 * builds; the matching intent filter lives in `src/debug/AndroidManifest.xml`.
 *
 * The main thread is not blocked: [goAsync] keeps the BroadcastReceiver
 * alive while a worker thread runs the model pipeline (each WAV takes
 * 30–60 s, well beyond the receiver's default 10 s budget).
 *
 * Host workflow (adb):
 *   1. adb push <wav> /data/local/tmp/<wavName>
 *   2. adb shell run-as com.aiglasses.memoryassistant.demo \
 *        sh -c 'mkdir -p files/eval_ali_input && \
 *               cp /data/local/tmp/<wavName> files/eval_ali_input/<wavName> && \
 *               rm /data/local/tmp/<wavName>'
 *   3. adb shell am broadcast \
 *        -a com.aiglasses.memoryassistant.debug.RUN_OFFLINE_EVAL \
 *        -p com.aiglasses.memoryassistant.demo \
 *        --es wav_name <wavName> --es case_id <caseId>
 *   4. adb shell run-as com.aiglasses.memoryassistant.demo \
 *        cat files/eval_ali_exports/<caseId>.json > out.json
 *
 * Result code: 0 = success (JSON written), 1 = failure (see logcat).
 */
class OfflineEvalDebugReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION) return
        val wavName = intent.getStringExtra(EXTRA_WAV_NAME)
        val caseId = intent.getStringExtra(EXTRA_CASE_ID)
        if (wavName.isNullOrBlank() || caseId.isNullOrBlank()) {
            Log.w(TAG, "missing wav_name or case_id extra")
            return
        }

        val pending = goAsync()
        Thread({
            try {
                val input = File(context.filesDir, "eval_ali_input/$wavName")
                require(input.isFile) { "missing input WAV at ${input.absolutePath}" }
                val uri = Uri.fromFile(input)
                val runner = OfflineAudioTestRunner(context)
                val result = runner.run(
                    uri = uri,
                    gainEnabled = false,
                    gainDecibels = OfflineAudioGain.MIN_DECIBELS,
                )
                val json = result.toEvalAliDebugJson(caseId)
                val outDir = File(context.filesDir, "eval_ali_exports")
                outDir.mkdirs()
                val out = File(outDir, "$caseId.json")
                out.writeText(json, Charsets.UTF_8)
                pending.setResultCode(0)
                Log.i(
                    TAG,
                    "ok $caseId segments=${result.segments.size} " +
                        "vad=${result.vadBackend} elapsed_ms=${result.elapsedMillis}",
                )
            } catch (t: Throwable) {
                Log.e(TAG, "failed $caseId", t)
                pending.setResultCode(1)
            } finally {
                pending.finish()
            }
        }, "offline-eval-debug-$caseId").start()
    }

    companion object {
        private const val ACTION = "com.aiglasses.memoryassistant.debug.RUN_OFFLINE_EVAL"
        private const val EXTRA_WAV_NAME = "wav_name"
        private const val EXTRA_CASE_ID = "case_id"
        private const val TAG = "OfflineEvalDebug"
    }
}