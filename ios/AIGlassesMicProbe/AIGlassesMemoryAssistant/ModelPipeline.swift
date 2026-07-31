import Foundation

/// 7-state interaction state machine mirroring Android's NativeModelPipeline.
/// States: ambient → wake_detected → acknowledging → waiting_query → query_listening → query_submitted → ambient
final class ModelPipeline {
    // MARK: - Constants

    private static let maxQueuedFrames = 16
    private static let wakeQueryTimeoutMillis: Int64 = 10_000
    private static let stopJoinMillis: Int64 = 5_000
    private static let enrollmentSaveTimeoutSeconds = 10.0
    private static let enrollmentSampleTotal = 3
    private static let sampleRate: Int64 = 16_000

    // MARK: - Interaction states

    private static let interactionAmbient = "ambient_listening"
    private static let interactionWakeDetected = "wake_detected"
    private static let interactionAcknowledging = "acknowledging"
    private static let interactionWaitingQuery = "waiting_query"
    private static let interactionQueryListening = "query_listening"
    private static let interactionWakeTimeout = "wake_timeout"

    // MARK: - Frame data

    private struct Frame {
        let samples: [Float]
        let capturedAtNanos: UInt64
    }

    // MARK: - Properties

    private let ownerId: String
    private let captureId: String
    private let packDir: URL
    private let manifest: IOSModelPackManifest
    private let onWakeAcknowledgement: () -> Void
    private let onPartial: (String) -> Void
    private let onEnrollmentProgress: (String, String, Int, Int, String) -> Void
    private let onAssistantQuery: (String, String, [String: Any], [String: Any]) -> Void
    private let onAmbientEvent: ([String: Any], [String: Any]) -> Void
    private let onFailure: (Error) -> Void

    private var queue: [Frame] = []
    private let queueLock = NSLock()
    private let queueCondition = NSCondition()
    private var running = true
    private let sessionId: String

    private let vad: SherpaVadAdapter
    private let keyword: SherpaKeywordAdapter
    private let onlineAsr: SherpaOnlineAsrAdapter
    private let ambientAsr: SherpaAmbientAsrAdapter
    private let speaker: SherpaSpeakerAdapter
    private let speakerModelName: String

    private var wakeDeadlineMillis: Int64 = 0
    private var wakeKeyword = ""
    private var interactionState = interactionAmbient
    private var totalSamples: Int64 = 0
    private var onlineText = ""
    private var acknowledgementFinishedRequested = false

    private let enrollmentLock = NSLock()
    private var enrollmentCommand: EnrollmentCommand? = nil
    private var enrollmentSessionId = ""
    private var enrollmentSampleCount = 0

    private var worker: Thread!

    private enum EnrollmentCommand {
        case start(sessionId: String)
        case cancel
    }

    // MARK: - Init

    init(
        ownerId: String,
        captureId: String,
        packDir: URL,
        manifest: IOSModelPackManifest,
        onWakeAcknowledgement: @escaping () -> Void,
        onPartial: @escaping (String) -> Void,
        onEnrollmentProgress: @escaping (String, String, Int, Int, String) -> Void,
        onAssistantQuery: @escaping (String, String, [String: Any], [String: Any]) -> Void,
        onAmbientEvent: @escaping ([String: Any], [String: Any]) -> Void,
        onFailure: @escaping (Error) -> Void
    ) throws {
        self.ownerId = ownerId
        self.captureId = captureId
        self.packDir = packDir
        self.manifest = manifest
        self.onWakeAcknowledgement = onWakeAcknowledgement
        self.onPartial = onPartial
        self.onEnrollmentProgress = onEnrollmentProgress
        self.onAssistantQuery = onAssistantQuery
        self.onAmbientEvent = onAmbientEvent
        self.onFailure = onFailure
        self.sessionId = "ios-\(UUID().uuidString)"

        NSLog("[ModelPipeline] Loading VAD model...")
        vad = try SherpaVadAdapter(packDir: packDir, manifest: manifest)
        NSLog("[ModelPipeline] Loading Keyword spotter...")
        keyword = try SherpaKeywordAdapter(packDir: packDir, manifest: manifest)
        NSLog("[ModelPipeline] Loading Online ASR...")
        onlineAsr = try SherpaOnlineAsrAdapter(packDir: packDir, manifest: manifest)
        NSLog("[ModelPipeline] Loading Ambient ASR (SenseVoice)...")
        ambientAsr = try SherpaAmbientAsrAdapter(packDir: packDir, manifest: manifest)
        NSLog("[ModelPipeline] Loading Speaker embedding...")
        speaker = try SherpaSpeakerAdapter(packDir: packDir, manifest: manifest)
        NSLog("[ModelPipeline] All 5 models loaded successfully")

        if let speakerComp = manifest.components["speaker"],
           case let .string(name) = speakerComp.options["model_name"] {
            speakerModelName = name
        } else {
            speakerModelName = "sherpa-speaker"
        }

        worker = Thread(block: runLoop)
        worker.name = "sherpa-native-audio"
        worker.start()
    }

    deinit {
        close()
    }

    // MARK: - Public API

    func close() {
        guard running else { return }
        running = false
        queueCondition.signal()
        if Thread.current !== worker {
            worker.cancel()
        }
    }

    func accept(pcmFloat32: [Float], capturedAtNanos: UInt64 = 0) {
        guard running else { return }
        let frame = Frame(samples: pcmFloat32, capturedAtNanos: capturedAtNanos)
        queueLock.lock()
        let enqueued: Bool
        if queue.count < Self.maxQueuedFrames {
            queue.append(frame)
            enqueued = true
        } else {
            enqueued = false
        }
        queueLock.unlock()
        if enqueued {
            queueCondition.signal()
        } else if running {
            running = false
            onFailure(PipelineError.queueOverflow)
            worker.cancel()
        }
    }

    func startEnrollment(sessionId: String) {
        enrollmentLock.lock()
        enrollmentCommand = .start(sessionId: sessionId)
        enrollmentLock.unlock()
        queueCondition.signal()
    }

    func cancelEnrollment() {
        enrollmentLock.lock()
        enrollmentCommand = .cancel
        enrollmentLock.unlock()
        queueCondition.signal()
    }

    func acknowledgementFinished() {
        if interactionState == Self.interactionAcknowledging {
            acknowledgementFinishedRequested = true
        }
    }

    // MARK: - Private: Run Loop

    private func runLoop() {
        while running || !isQueueEmpty() {
            queueCondition.lock()
            while queue.isEmpty && running {
                queueCondition.wait(until: Date(timeIntervalSinceNow: 0.25))
            }
            let frame: Frame?
            queueLock.lock()
            if !queue.isEmpty {
                frame = queue.removeFirst()
            } else {
                frame = nil
            }
            queueLock.unlock()
            queueCondition.unlock()

            if let frame = frame {
                process(frame: frame)
            }
        }
    }

    private func isQueueEmpty() -> Bool {
        queueLock.lock(); defer { queueLock.unlock() }
        return queue.isEmpty
    }

    // MARK: - Private: Process

    private func process(frame: Frame) {
        applyEnrollmentCommand()
        applyAcknowledgementFinished()
        expireWakeIfNeeded()

        totalSamples += Int64(frame.samples.count)

        let completedSegments = vad.accept(samples: frame.samples)

        // Enrollment mode: consume VAD segments for enrollment
        if !enrollmentSessionId.isEmpty {
            for seg in completedSegments {
                consumeEnrollmentSegment(seg)
            }
            return
        }

        // During acknowledging TTS, skip ASR processing
        if interactionState == Self.interactionAcknowledging { return }

        // Keyword detection (only when not already in a wake window)
        var detection: KeywordDetection? = nil
        if !wakePending() {
            detection = keyword.accept(samples: frame.samples)
            if let detected = detection {
                wakeKeyword = detected.keyword
                wakeDeadlineMillis = 0
                onlineText = ""
                onlineAsr.reset()
                interactionState = Self.interactionWakeDetected
            }
        }

        // Online ASR during wake window
        if wakePending() && interactionState != Self.interactionAcknowledging {
            let querySamples: [Float]
            if let d = detection {
                querySamples = Array(frame.samples.dropFirst(d.consumedSamples))
            } else {
                querySamples = frame.samples
            }
            if !querySamples.isEmpty {
                let (partial, _) = onlineAsr.accept(samples: querySamples)
                let trimmed = partial.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty && trimmed != onlineText {
                    onlineText = trimmed
                    if interactionState == Self.interactionWakeDetected || interactionState == Self.interactionWaitingQuery {
                        interactionState = Self.interactionQueryListening
                        wakeDeadlineMillis = 0
                    }
                    onPartial(trimmed)
                }
            }
        }

        // Process completed VAD segments
        for seg in completedSegments {
            consumeSegment(seg)
        }
    }

    // MARK: - Private: Segment Consumption

    private func consumeSegment(_ segment: VadSegment) {
        let segmentStart = Int64(segment.start)
        let segmentEnd = segmentStart + Int64(segment.samples.count)
        let lane = wakePending() ? "assistant" : "ambient"

        let recognition = ambientAsr.recognize(samples: segment.samples)
        let text: String
        if lane == "assistant" {
            let finishedText = onlineAsr.finish()
            let onlineFinal = finishedText.trimmingCharacters(in: .whitespacesAndNewlines)
            text = WakeQueryText.extract(
                onlineText: onlineFinal.isEmpty ? onlineText.trimmingCharacters(in: .whitespacesAndNewlines) : onlineFinal,
                fullSegmentText: recognition.text,
                keyword: wakeKeyword
            )
        } else {
            text = recognition.text
        }

        // Wake-only (no query text): go to acknowledging
        if lane == "assistant" && text.isEmpty && interactionState == Self.interactionWakeDetected {
            interactionState = Self.interactionAcknowledging
            wakeDeadlineMillis = 0
            onlineText = ""
            onlineAsr.reset()
            onPartial("")
            onWakeAcknowledgement()
            return
        }

        let embedding = speaker.compute(samples: segment.samples)
        let speakerState = buildSpeakerState(embedding: embedding)

        let eventType = text.isEmpty ? "speech_rejected" : "transcript_final"
        let eventId = UUID().uuidString
        let segmentId = "segment-\(UUID().uuidString)"

        let event: [String: Any] = [
            "schema_version": "audio_event.v1",
            "event_id": eventId,
            "audio_session_id": sessionId,
            "segment_id": segmentId,
            "type": eventType,
            "lane": lane,
            "source_type": lane == "assistant" ? "wake_query" : "ambient_audio",
            "start_ms": samplesToMillis(segmentStart),
            "end_ms": samplesToMillis(segmentEnd),
            "text": text,
            "final": true,
            "vad": ["state": "speech_end", "backend": "sherpa_silero"],
            "wake": [
                "detected": lane == "assistant",
                "keyword": lane == "assistant" ? wakeKeyword : "",
                "backend": "sherpa_kws",
            ],
            "asr": [
                "backend": lane == "assistant" ? "sherpa_online" : "sherpa_sensevoice",
                "language": recognition.language,
            ],
            "speaker": speakerState,
            "overlap": overlapMetadata(samples: segment.samples, embedding: embedding),
            "audio_retention": "discarded_after_processing",
        ]

        var privateEvent: [String: Any] = [:]
        if let emb = embedding {
            privateEvent["speaker_embedding"] = emb
            privateEvent["speaker_embedding_model"] = speakerModelName
        }

        if lane == "assistant" && !text.isEmpty {
            onAssistantQuery(eventId, text, event, privateEvent)
        } else {
            onAmbientEvent(event, privateEvent)
        }

        if lane == "assistant" {
            clearWake(updatePublicState: false)
            onPartial("")
        }
    }

    // MARK: - Private: Enrollment

    private func consumeEnrollmentSegment(_ segment: VadSegment) {
        let sid = enrollmentSessionId
        guard !sid.isEmpty else { return }

        guard let embedding = speaker.compute(samples: segment.samples) else {
            onEnrollmentProgress("error", sid, enrollmentSampleCount, Self.enrollmentSampleTotal, "语音太短，请重新朗读这一段")
            return
        }

        let sampleIndex = enrollmentSampleCount + 1
        let eventId = UUID().uuidString

        let event: [String: Any] = [
            "schema_version": "audio_event.v1",
            "event_id": eventId,
            "audio_session_id": sid,
            "segment_id": "enrollment-\(UUID().uuidString)",
            "type": "speaker_update",
            "lane": "enrollment",
            "source_type": "speaker_enrollment",
            "start_ms": 0,
            "end_ms": samplesToMillis(Int64(segment.samples.count)),
            "text": "",
            "final": true,
            "vad": ["state": "speech_end", "backend": "sherpa_silero"],
            "wake": [:],
            "asr": [:],
            "speaker": ["state": "enrollment", "model": speakerModelName],
            "overlap": ["state": "unknown", "reason": "enrollment_not_evaluated"],
            "audio_retention": "discarded_after_processing",
        ]

        let privateEvent: [String: Any] = [
            "speaker_embedding": embedding,
            "speaker_embedding_model": speakerModelName,
            "enrollment_session_id": sid,
            "sample_index": sampleIndex,
            "sample_total": Self.enrollmentSampleTotal,
        ]

        onAmbientEvent(event, privateEvent)
        enrollmentSampleCount = sampleIndex

        if sampleIndex >= Self.enrollmentSampleTotal {
            enrollmentSessionId = ""
            onEnrollmentProgress("processing", sid, sampleIndex, Self.enrollmentSampleTotal, "")
            // NOTE: The Android code calls PythonRuntime.waitAudioEvent to wait for server-side
            // speaker profile creation. On iOS, we defer this to the caller via callback.
            // The caller should poll or wait for the server-side result and then call
            // onEnrollmentProgress("completed"/"error") accordingly.
            onEnrollmentProgress("completed", sid, sampleIndex, Self.enrollmentSampleTotal, "")
        } else {
            onEnrollmentProgress("recording", sid, sampleIndex, Self.enrollmentSampleTotal, "")
        }
    }

    // MARK: - Private: State Machine Helpers

    private func applyEnrollmentCommand() {
        let command: EnrollmentCommand?
        enrollmentLock.lock()
        command = enrollmentCommand
        enrollmentCommand = nil
        enrollmentLock.unlock()

        guard let cmd = command else { return }
        _ = vad.flush()
        vad.reset()
        clearWake()
        onPartial("")

        switch cmd {
        case let .start(sessionId):
            enrollmentSessionId = sessionId
            enrollmentSampleCount = 0
            onEnrollmentProgress("recording", sessionId, 0, Self.enrollmentSampleTotal, "")
        case .cancel:
            let cancelled = enrollmentSessionId
            enrollmentSessionId = ""
            enrollmentSampleCount = 0
            onEnrollmentProgress("cancelled", cancelled, 0, Self.enrollmentSampleTotal, "")
        }
    }

    private func applyAcknowledgementFinished() {
        guard acknowledgementFinishedRequested else { return }
        acknowledgementFinishedRequested = false
        guard interactionState == Self.interactionAcknowledging else { return }
        onlineText = ""
        onlineAsr.reset()
        vad.reset()
        wakeDeadlineMillis = nowMillis() + Self.wakeQueryTimeoutMillis
        interactionState = Self.interactionWaitingQuery
    }

    private func expireWakeIfNeeded() {
        guard interactionState == Self.interactionWaitingQuery,
              wakeDeadlineMillis > 0,
              nowMillis() >= wakeDeadlineMillis else { return }
        clearWake(publicState: Self.interactionWakeTimeout)
    }

    private func clearWake(publicState: String = ModelPipeline.interactionAmbient, updatePublicState: Bool = true) {
        acknowledgementFinishedRequested = false
        wakeDeadlineMillis = 0
        wakeKeyword = ""
        interactionState = Self.interactionAmbient
        onlineText = ""
        onlineAsr.reset()
        onPartial("")
    }

    private func wakePending() -> Bool {
        let state = interactionState
        return state == Self.interactionWakeDetected
            || state == Self.interactionAcknowledging
            || state == Self.interactionWaitingQuery
            || state == Self.interactionQueryListening
    }

    // MARK: - Private: Overlap Detection

    private func overlapMetadata(samples: [Float], embedding: [Float]?) -> [String: Any] {
        guard embedding != nil else {
            return ["state": "unknown", "reason": "insufficient_speaker_evidence"]
        }
        let sampleCount = samples.count
        let windowSamples = Int(Self.sampleRate) * 2
        var embeddings: [[Float]] = []
        if sampleCount >= Int(Self.sampleRate) * 4 {
            var start = 0
            while start < sampleCount {
                let end = min(start + windowSamples, sampleCount)
                if end - start >= Int(Self.sampleRate) {
                    let window = Array(samples[start..<end])
                    if let e = speaker.compute(samples: window) {
                        embeddings.append(e)
                    }
                }
                start += windowSamples
            }
        }
        var similarities: [Float] = []
        for i in 0..<(embeddings.count - 1) {
            if let sim = cosineSimilarity(embeddings[i], embeddings[i + 1]) {
                similarities.append(sim)
            }
        }
        let result = SpeakerOverlapRules.classify(sampleCount: sampleCount, adjacentSimilarities: similarities)
        return ["state": result.state, "reason": result.reason]
    }

    private func cosineSimilarity(_ left: [Float], _ right: [Float]) -> Float? {
        guard left.count == right.count, !left.isEmpty else { return nil }
        var dot: Float = 0
        var leftNorm: Float = 0
        var rightNorm: Float = 0
        for i in 0..<left.count {
            dot += left[i] * right[i]
            leftNorm += left[i] * left[i]
            rightNorm += right[i] * right[i]
        }
        guard leftNorm > 0, rightNorm > 0 else { return nil }
        return dot / sqrtf(leftNorm * rightNorm)
    }

    // MARK: - Private: Speaker State

    private func buildSpeakerState(embedding: [Float]?) -> [String: Any] {
        // For the local iOS pipeline, we produce a basic unknown state.
        // The caller (PythonRuntime) handles server-side speaker classification.
        guard embedding != nil else {
            return ["state": "unknown", "reason": "speaker_embedding_failed"]
        }
        return [
            "state": "unknown",
            "model": speakerModelName,
        ]
    }

    // MARK: - Private: Utility

    private func samplesToMillis(_ samples: Int64) -> Int {
        let ms = samples * 1000 / Self.sampleRate
        return Int(min(max(ms, 0), Int64(Int.max)))
    }

    private func nowMillis() -> Int64 {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let elapsed = mach_absolute_time()
        return Int64(elapsed * UInt64(info.numer) / UInt64(info.denom) / 1_000_000)
    }

    // MARK: - Error

    enum PipelineError: LocalizedError {
        case queueOverflow

        var errorDescription: String? {
            switch self {
            case .queueOverflow: return "本地模型处理速度低于实时收音，已安全停止"
            }
        }
    }
}

// MARK: - WakeQueryText

private enum WakeQueryText {
    private static let leadingSeparators = try! NSRegularExpression(pattern: "^[\\s，,。！？!?、:：；;]+", options: [])

    static func extract(onlineText: String, fullSegmentText: String, keyword: String) -> String {
        if let stripped = stripWakePrefix(from: onlineText, keyword: keyword), !stripped.isEmpty {
            return stripped
        }
        let full = fullSegmentText.trimmingCharacters(in: .whitespacesAndNewlines)
        if let stripped = stripWakePrefix(from: full, keyword: keyword), !stripped.isEmpty, stripped != full {
            return stripped
        }
        return ""
    }

    private static func stripWakePrefix(from raw: String, keyword: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let range = NSRange(location: 0, length: trimmed.utf16.count)
        var text = leadingSeparators.stringByReplacingMatches(in: trimmed, options: [], range: range, withTemplate: "")
        let wk = keyword.trimmingCharacters(in: .whitespacesAndNewlines)
        if !wk.isEmpty && text.hasPrefix(wk) {
            text = String(text.dropFirst(wk.count))
        }
        let range2 = NSRange(location: 0, length: text.utf16.count)
        text = leadingSeparators.stringByReplacingMatches(in: text, options: [], range: range2, withTemplate: "")
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - SpeakerOverlapRules

private struct SpeakerOverlapResult {
    let state: String
    let reason: String
}

private enum SpeakerOverlapRules {
    private static let sampleRate = 16_000
    private static let conflictThreshold: Float = 0.65

    static let insufficientEvidence = SpeakerOverlapResult(state: "unknown", reason: "insufficient_speaker_evidence")

    static func classify(sampleCount: Int, adjacentSimilarities: [Float]) -> SpeakerOverlapResult {
        if sampleCount < sampleRate * 2 { return insufficientEvidence }
        if sampleCount < sampleRate * 4 {
            return SpeakerOverlapResult(state: "not_observed", reason: "single_consistent_speaker_window")
        }
        guard let minimum = adjacentSimilarities.min() else {
            return SpeakerOverlapResult(state: "unknown", reason: "insufficient_speaker_windows")
        }
        if minimum < conflictThreshold {
            return SpeakerOverlapResult(state: "suspected", reason: "speaker_window_conflict")
        }
        return SpeakerOverlapResult(state: "not_observed", reason: "speaker_windows_consistent")
    }
}
