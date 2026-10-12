package com.aiglasses.memoryassistant.demo

import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject

/** Opt-in, bounded in-memory acceptance markers; never retain text or audio. */
object AcceptanceTiming {
    @Volatile var enabled = false
        private set
    private val events = ArrayDeque<JSONObject>()
    fun clockMs(): Double = SystemClock.elapsedRealtimeNanos() / 1_000_000.0

    @Synchronized fun enable(value: Boolean) { enabled = value; events.clear() }
    @Synchronized fun mark(phase: String, ids: JSONObject = JSONObject()) {
        if (!enabled) return
        val event = JSONObject().put("phase", phase).put("clock", "android_elapsed")
            .put("at_ms", clockMs())
        for (key in listOf("interaction_id", "turn_id", "event_id", "job_id", "utterance_id")) {
            ids.optString(key).takeIf(String::isNotBlank)?.let { event.put(key, it.take(128)) }
        }
        events.addLast(event)
        if (events.size > 1000) events.removeFirst()
    }
    @Synchronized fun snapshot(): JSONArray = JSONArray(events.map { JSONObject(it.toString()) })
    @Synchronized fun drain(): JSONArray { val result = snapshot(); events.clear(); return result }
}
