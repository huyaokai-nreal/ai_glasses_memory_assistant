package com.aiglasses.memoryassistant.demo

import kotlin.math.sqrt

/**
 * Keeps anonymous speaker centroids only while one NativeModelPipeline lives.
 * A track is deliberately not an identity and must not cross capture boundaries.
 */
internal class CaptureSpeakerTracker(
    private val matchThreshold: Float = 0.78f,
) {
    internal data class Assignment(
        val voiceGroup: String = "",
        val confidence: Float? = null,
        val scope: String = "capture",
        val reason: String = "not_assigned",
    )

    private data class Track(val centroid: FloatArray, val sampleCount: Int)

    private val tracks = mutableListOf<Track>()

    fun assign(
        embedding: FloatArray?,
        speakerState: String,
        overlapState: String,
    ): Assignment {
        if (speakerState == "user" || speakerState == "self") {
            return Assignment(reason = "self_not_anonymous")
        }
        if (overlapState != "not_observed") {
            return Assignment(reason = "overlap_or_insufficient_evidence")
        }
        val vector = normalizedCopy(embedding) ?: return Assignment(reason = "embedding_unavailable")
        val scores = tracks.map { cosineSimilarity(vector, it.centroid) ?: -1f }
        val bestIndex = scores.indices.maxByOrNull { scores[it] }
        val bestScore = bestIndex?.let { scores[it] } ?: -1f
        val index = if (bestIndex != null && bestScore >= matchThreshold) {
            tracks[bestIndex] = merge(tracks[bestIndex], vector)
            bestIndex
        } else {
            tracks += Track(vector, 1)
            tracks.lastIndex
        }
        return Assignment(
            voiceGroup = "spk_%02d".format(index + 1),
            confidence = if (bestScore >= 0f) bestScore else 1f,
            reason = if (bestIndex == null || bestScore < matchThreshold) "new_capture_track" else "matched_capture_track",
        )
    }

    fun reset() = tracks.clear()

    private fun merge(track: Track, vector: FloatArray): Track {
        val merged = FloatArray(vector.size) { index ->
            (track.centroid[index] * track.sampleCount + vector[index]) / (track.sampleCount + 1)
        }
        return Track(normalizedCopy(merged) ?: track.centroid, track.sampleCount + 1)
    }

    private fun normalizedCopy(value: FloatArray?): FloatArray? {
        if (value == null || value.isEmpty() || value.any { !it.isFinite() }) return null
        val norm = sqrt(value.fold(0.0) { total, item ->
            total + item.toDouble() * item.toDouble()
        }).toFloat()
        if (norm <= 0f) return null
        return FloatArray(value.size) { index -> value[index] / norm }
    }

    private fun cosineSimilarity(left: FloatArray, right: FloatArray): Float? {
        if (left.size != right.size || left.isEmpty()) return null
        return left.indices.sumOf { index -> (left[index] * right[index]).toDouble() }.toFloat()
    }
}
