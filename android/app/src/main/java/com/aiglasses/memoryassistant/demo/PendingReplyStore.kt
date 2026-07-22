package com.aiglasses.memoryassistant.demo

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class PendingReplyStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun markPending(eventId: String, query: String) {
        synchronized(lock) {
            if (eventId.isBlank()) return
            val bounded = (pendingIdsLocked() - eventId).toList()
                .takeLast(MAX_PENDING_IDS - 1)
                .toSet() + eventId
            val previousQueries = pendingQueriesLocked()
            val queries = JSONObject()
            bounded.forEach { pendingId ->
                queries.put(
                    pendingId,
                    if (pendingId == eventId) query.take(MAX_QUERY_CHARS) else previousQueries.optString(pendingId),
                )
            }
            prefs.edit()
                .putStringSet(KEY_PENDING, bounded)
                .putString(KEY_PENDING_QUERIES, queries.toString())
                .apply()
        }
    }

    fun pendingIds(): Set<String> = synchronized(lock) { pendingIdsLocked() }

    fun markCompleted(eventId: String, reply: String) {
        synchronized(lock) {
            val pending = pendingIdsLocked() - eventId
            val queries = pendingQueriesLocked()
            val query = queries.optString(eventId)
            queries.remove(eventId)
            val existing = completedRepliesLocked()
            val completed = JSONArray()
            val start = (existing.length() - (MAX_COMPLETED_REPLIES - 1)).coerceAtLeast(0)
            for (index in start until existing.length()) completed.put(existing.get(index))
            completed.put(
                JSONObject()
                    .put("event_id", eventId)
                    .put("query", query)
                    .put("reply", reply.take(MAX_REPLY_CHARS)),
            )
            prefs.edit()
                .putStringSet(KEY_PENDING, pending)
                .putString(KEY_PENDING_QUERIES, queries.toString())
                .putString(KEY_COMPLETED, completed.toString())
                .apply()
        }
    }

    fun removePending(eventId: String) {
        synchronized(lock) {
            val queries = pendingQueriesLocked()
            queries.remove(eventId)
            prefs.edit()
                .putStringSet(KEY_PENDING, pendingIdsLocked() - eventId)
                .putString(KEY_PENDING_QUERIES, queries.toString())
                .apply()
        }
    }

    fun consumeCompleted(): String {
        return synchronized(lock) {
            val completed = completedRepliesLocked()
            prefs.edit().remove(KEY_COMPLETED).apply()
            completed.toString()
        }
    }

    private fun pendingIdsLocked(): Set<String> = prefs.getStringSet(KEY_PENDING, emptySet()).orEmpty().toSet()

    private fun completedRepliesLocked(): JSONArray = runCatching {
        JSONArray(prefs.getString(KEY_COMPLETED, "[]"))
    }.getOrDefault(JSONArray())

    private fun pendingQueriesLocked(): JSONObject = runCatching {
        JSONObject(prefs.getString(KEY_PENDING_QUERIES, "{}").orEmpty())
    }.getOrDefault(JSONObject())

    companion object {
        private const val PREFS_NAME = "pending_device_replies"
        private const val KEY_PENDING = "pending_event_ids"
        private const val KEY_COMPLETED = "completed_replies"
        private const val KEY_PENDING_QUERIES = "pending_queries"
        private const val MAX_QUERY_CHARS = 2_000
        private const val MAX_REPLY_CHARS = 32_000
        private const val MAX_PENDING_IDS = 200
        private const val MAX_COMPLETED_REPLIES = 20
        private val lock = Any()
    }
}
