import Foundation
import SherpaOnnxC

// MARK: - Data types

struct KeywordDetection {
    let keyword: String
    let consumedSamples: Int
}

struct AmbientRecognition {
    let text: String
    let language: String
    let emotion: String
    let acousticEvent: String
}

struct VadSegment {
    let start: Int
    let samples: [Float]
}

// MARK: - C String Helpers

/// strdup a Swift string and return it as const char * (UnsafePointer<CChar>?).
private func cstr(_ s: String) -> UnsafePointer<CChar>? {
    strdup(s).map { UnsafePointer($0) }
}

/// free a const char * that was allocated with strdup.
private func cstrFree(_ ptr: UnsafePointer<CChar>?) {
    if let p = ptr { free(UnsafeMutablePointer(mutating: p)) }
}

// MARK: - SherpaVadAdapter

final class SherpaVadAdapter {
    private let vad: OpaquePointer

    init(packDir: URL, manifest: IOSModelPackManifest) throws {
        guard let component = manifest.components["vad"] else {
            throw SherpaAdapterError.missingComponent("vad")
        }
        guard component.engine == "silero_vad" else {
            throw SherpaAdapterError.unsupportedEngine("vad", component.engine)
        }
        guard let modelRole = component.roles["model"] else {
            throw SherpaAdapterError.missingRole("vad", "model")
        }

        var config = SherpaOnnxVadModelConfig()
        memset(&config, 0, MemoryLayout<SherpaOnnxVadModelConfig>.size)
        config.silero_vad.model = cstr(packDir.appendingPathComponent(modelRole).path)
        config.silero_vad.threshold = component.floatOption("threshold", fallback: 0.5)
        config.silero_vad.min_silence_duration = component.floatOption("min_silence_seconds", fallback: 0.5)
        config.silero_vad.min_speech_duration = component.floatOption("min_speech_seconds", fallback: 0.25)
        config.silero_vad.max_speech_duration = component.floatOption("max_speech_seconds", fallback: 30.0)
        config.silero_vad.window_size = Int32(component.intOption("window_size", fallback: 512))
        config.sample_rate = SherpaConstants.sampleRate
        config.num_threads = Int32(max(component.intOption("num_threads", fallback: 1), 1))
        config.provider = cstr("cpu")
        config.debug = 0
        defer {
            cstrFree(config.silero_vad.model)
            cstrFree(config.provider)
        }
        guard let detector = SherpaOnnxCreateVoiceActivityDetector(&config, 30.0) else {
            throw SherpaAdapterError.creationFailed("VAD")
        }
        vad = detector
    }

    func accept(samples: [Float]) -> [VadSegment] {
        samples.withUnsafeBufferPointer { ptr in
            SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, ptr.baseAddress, Int32(ptr.count))
        }
        return drainSegments()
    }

    func flush() -> [VadSegment] {
        SherpaOnnxVoiceActivityDetectorFlush(vad)
        return drainSegments()
    }

    func reset() {
        SherpaOnnxVoiceActivityDetectorReset(vad)
    }

    private func drainSegments() -> [VadSegment] {
        var segments: [VadSegment] = []
        while SherpaOnnxVoiceActivityDetectorEmpty(vad) == 0 {
            if let segPtr = SherpaOnnxVoiceActivityDetectorFront(vad) {
                let seg = segPtr.pointee
                let samplesCopy = seg.samples.map { Array(UnsafeBufferPointer(start: $0, count: Int(seg.n))) } ?? []
                segments.append(VadSegment(start: Int(seg.start), samples: samplesCopy))
                SherpaOnnxDestroySpeechSegment(segPtr)
            }
            SherpaOnnxVoiceActivityDetectorPop(vad)
        }
        return segments
    }

    deinit {
        SherpaOnnxDestroyVoiceActivityDetector(vad)
    }
}

// MARK: - SherpaKeywordAdapter

final class SherpaKeywordAdapter {
    private let spotter: OpaquePointer
    private var stream: OpaquePointer
    private let detectionChunkSamples = 320

    init(packDir: URL, manifest: IOSModelPackManifest) throws {
        guard let component = manifest.components["kws"] else {
            throw SherpaAdapterError.missingComponent("kws")
        }
        guard let encoderRole = component.roles["encoder"],
              let decoderRole = component.roles["decoder"],
              let joinerRole = component.roles["joiner"],
              let tokensRole = component.roles["tokens"],
              let keywordsRole = component.roles["keywords"] else {
            throw SherpaAdapterError.missingRole("kws", "encoder/decoder/joiner/tokens/keywords")
        }

        var config = SherpaOnnxKeywordSpotterConfig()
        memset(&config, 0, MemoryLayout<SherpaOnnxKeywordSpotterConfig>.size)
        config.feat_config = SherpaOnnxFeatureConfig(sample_rate: SherpaConstants.sampleRate, feature_dim: 80)
        config.model_config = makeOnlineModelConfig(
            packDir: packDir, component: component,
            encoderRole: encoderRole, decoderRole: decoderRole,
            joinerRole: joinerRole, tokensRole: tokensRole
        )
        config.keywords_file = cstr(packDir.appendingPathComponent(keywordsRole).path)
        config.keywords_score = component.floatOption("keywords_score", fallback: 1.5)
        config.keywords_threshold = component.floatOption("keywords_threshold", fallback: 0.25)
        config.num_trailing_blanks = Int32(component.intOption("num_trailing_blanks", fallback: 1))
        config.max_active_paths = Int32(component.intOption("max_active_paths", fallback: 4))
        defer {
            cstrFree(config.keywords_file)
            freeOnlineModelConfig(&config.model_config)
        }

        guard let s = SherpaOnnxCreateKeywordSpotter(&config) else {
            throw SherpaAdapterError.creationFailed("KeywordSpotter")
        }
        spotter = s
        guard let str = SherpaOnnxCreateKeywordStream(spotter) else {
            SherpaOnnxDestroyKeywordSpotter(spotter)
            throw SherpaAdapterError.creationFailed("KeywordStream")
        }
        stream = str
    }

    func accept(samples: [Float]) -> KeywordDetection? {
        var consumed = 0
        while consumed < samples.count {
            let end = min(consumed + detectionChunkSamples, samples.count)
            let chunk = Array(samples[consumed..<end])
            chunk.withUnsafeBufferPointer { ptr in
                SherpaOnnxOnlineStreamAcceptWaveform(stream, SherpaConstants.sampleRate, ptr.baseAddress, Int32(ptr.count))
            }
            while SherpaOnnxIsKeywordStreamReady(spotter, stream) == 1 {
                SherpaOnnxDecodeKeywordStream(spotter, stream)
            }
            consumed = end
            if let resultPtr = SherpaOnnxGetKeywordResult(spotter, stream) {
                let keyword = String(cString: resultPtr.pointee.keyword).trimmingCharacters(in: .whitespacesAndNewlines)
                SherpaOnnxDestroyKeywordResult(resultPtr)
                if !keyword.isEmpty {
                    SherpaOnnxResetKeywordStream(spotter, stream)
                    return KeywordDetection(keyword: keyword, consumedSamples: consumed)
                }
            }
        }
        return nil
    }

    deinit {
        SherpaOnnxDestroyOnlineStream(stream)
        SherpaOnnxDestroyKeywordSpotter(spotter)
    }
}

// MARK: - SherpaOnlineAsrAdapter

final class SherpaOnlineAsrAdapter {
    private let recognizer: OpaquePointer
    private var stream: OpaquePointer

    init(packDir: URL, manifest: IOSModelPackManifest) throws {
        guard let component = manifest.components["online_asr"] else {
            throw SherpaAdapterError.missingComponent("online_asr")
        }
        guard let encoderRole = component.roles["encoder"],
              let decoderRole = component.roles["decoder"],
              let joinerRole = component.roles["joiner"],
              let tokensRole = component.roles["tokens"] else {
            throw SherpaAdapterError.missingRole("online_asr", "encoder/decoder/joiner/tokens")
        }

        var config = SherpaOnnxOnlineRecognizerConfig()
        memset(&config, 0, MemoryLayout<SherpaOnnxOnlineRecognizerConfig>.size)
        config.feat_config = SherpaOnnxFeatureConfig(sample_rate: SherpaConstants.sampleRate, feature_dim: 80)
        config.model_config = makeOnlineModelConfig(
            packDir: packDir, component: component,
            encoderRole: encoderRole, decoderRole: decoderRole,
            joinerRole: joinerRole, tokensRole: tokensRole
        )
        config.enable_endpoint = 1
        config.decoding_method = cstr("greedy_search")
        config.max_active_paths = Int32(component.intOption("max_active_paths", fallback: 4))
        defer {
            cstrFree(config.decoding_method)
            freeOnlineModelConfig(&config.model_config)
        }
        guard let r = SherpaOnnxCreateOnlineRecognizer(&config) else {
            throw SherpaAdapterError.creationFailed("OnlineRecognizer")
        }
        recognizer = r
        guard let s = SherpaOnnxCreateOnlineStream(recognizer) else {
            SherpaOnnxDestroyOnlineRecognizer(recognizer)
            throw SherpaAdapterError.creationFailed("OnlineStream")
        }
        stream = s
    }

    /// Returns (partialText, isEndpoint)
    func accept(samples: [Float]) -> (String, Bool) {
        samples.withUnsafeBufferPointer { ptr in
            SherpaOnnxOnlineStreamAcceptWaveform(stream, SherpaConstants.sampleRate, ptr.baseAddress, Int32(ptr.count))
        }
        while SherpaOnnxIsOnlineStreamReady(recognizer, stream) == 1 {
            SherpaOnnxDecodeOnlineStream(recognizer, stream)
        }
        let isEnd = SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream) == 1
        guard let resultPtr = SherpaOnnxGetOnlineStreamResult(recognizer, stream) else {
            return ("", isEnd)
        }
        let text = String(cString: resultPtr.pointee.text).trimmingCharacters(in: .whitespacesAndNewlines)
        SherpaOnnxDestroyOnlineRecognizerResult(resultPtr)
        return (text, isEnd)
    }

    func finish() -> String {
        SherpaOnnxOnlineStreamInputFinished(stream)
        while SherpaOnnxIsOnlineStreamReady(recognizer, stream) == 1 {
            SherpaOnnxDecodeOnlineStream(recognizer, stream)
        }
        guard let resultPtr = SherpaOnnxGetOnlineStreamResult(recognizer, stream) else {
            return ""
        }
        let text = String(cString: resultPtr.pointee.text).trimmingCharacters(in: .whitespacesAndNewlines)
        SherpaOnnxDestroyOnlineRecognizerResult(resultPtr)
        return text
    }

    func reset() {
        SherpaOnnxOnlineStreamReset(recognizer, stream)
    }

    deinit {
        SherpaOnnxDestroyOnlineStream(stream)
        SherpaOnnxDestroyOnlineRecognizer(recognizer)
    }
}

// MARK: - SherpaAmbientAsrAdapter

final class SherpaAmbientAsrAdapter {
    private let recognizer: OpaquePointer

    init(packDir: URL, manifest: IOSModelPackManifest) throws {
        guard let component = manifest.components["ambient_asr"] else {
            throw SherpaAdapterError.missingComponent("ambient_asr")
        }
        guard component.engine == "sense_voice" else {
            throw SherpaAdapterError.unsupportedEngine("ambient_asr", component.engine)
        }
        guard let modelRole = component.roles["model"],
              let tokensRole = component.roles["tokens"] else {
            throw SherpaAdapterError.missingRole("ambient_asr", "model/tokens")
        }

        var config = SherpaOnnxOfflineRecognizerConfig()
        memset(&config, 0, MemoryLayout<SherpaOnnxOfflineRecognizerConfig>.size)
        config.feat_config = SherpaOnnxFeatureConfig(sample_rate: SherpaConstants.sampleRate, feature_dim: 80)

        config.model_config.sense_voice.model = cstr(packDir.appendingPathComponent(modelRole).path)
        config.model_config.sense_voice.language = cstr(component.stringOption("language", fallback: "auto"))
        config.model_config.sense_voice.use_itn = component.boolOption("use_itn", fallback: true) ? 1 : 0
        config.model_config.tokens = cstr(packDir.appendingPathComponent(tokensRole).path)
        config.model_config.num_threads = Int32(max(component.intOption("num_threads", fallback: 2), 1))
        config.model_config.provider = cstr("cpu")
        config.model_config.debug = 0
        config.decoding_method = cstr("greedy_search")
        defer {
            cstrFree(config.model_config.sense_voice.model)
            cstrFree(config.model_config.sense_voice.language)
            cstrFree(config.model_config.tokens)
            cstrFree(config.model_config.provider)
            cstrFree(config.decoding_method)
        }
        guard let r = SherpaOnnxCreateOfflineRecognizer(&config) else {
            throw SherpaAdapterError.creationFailed("OfflineRecognizer(SenseVoice)")
        }
        recognizer = r
    }

    func recognize(samples: [Float]) -> AmbientRecognition {
        guard let stream = SherpaOnnxCreateOfflineStream(recognizer) else {
            return AmbientRecognition(text: "", language: "", emotion: "", acousticEvent: "")
        }
        defer { SherpaOnnxDestroyOfflineStream(stream) }
        samples.withUnsafeBufferPointer { ptr in
            SherpaOnnxAcceptWaveformOffline(stream, SherpaConstants.sampleRate, ptr.baseAddress, Int32(ptr.count))
        }
        SherpaOnnxDecodeOfflineStream(recognizer, stream)
        guard let resultPtr = SherpaOnnxGetOfflineStreamResult(stream) else {
            return AmbientRecognition(text: "", language: "", emotion: "", acousticEvent: "")
        }
        defer { SherpaOnnxDestroyOfflineRecognizerResult(resultPtr) }
        return AmbientRecognition(
            text: String(cString: resultPtr.pointee.text).trimmingCharacters(in: .whitespacesAndNewlines),
            language: resultPtr.pointee.lang.map { String(cString: $0) } ?? "",
            emotion: resultPtr.pointee.emotion.map { String(cString: $0) } ?? "",
            acousticEvent: resultPtr.pointee.event.map { String(cString: $0) } ?? ""
        )
    }

    deinit {
        SherpaOnnxDestroyOfflineRecognizer(recognizer)
    }
}

// MARK: - SherpaSpeakerAdapter

final class SherpaSpeakerAdapter {
    private let extractor: OpaquePointer
    private let dim: Int32

    init(packDir: URL, manifest: IOSModelPackManifest) throws {
        guard let component = manifest.components["speaker"] else {
            throw SherpaAdapterError.missingComponent("speaker")
        }
        guard component.engine == "speaker_embedding" else {
            throw SherpaAdapterError.unsupportedEngine("speaker", component.engine)
        }
        guard let modelRole = component.roles["model"] else {
            throw SherpaAdapterError.missingRole("speaker", "model")
        }

        var config = SherpaOnnxSpeakerEmbeddingExtractorConfig()
        memset(&config, 0, MemoryLayout<SherpaOnnxSpeakerEmbeddingExtractorConfig>.size)
        config.model = cstr(packDir.appendingPathComponent(modelRole).path)
        config.num_threads = Int32(max(component.intOption("num_threads", fallback: 1), 1))
        config.debug = 0
        config.provider = cstr("cpu")
        defer {
            cstrFree(config.model)
            cstrFree(config.provider)
        }
        guard let e = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config) else {
            throw SherpaAdapterError.creationFailed("SpeakerEmbeddingExtractor")
        }
        extractor = e
        dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor)
    }

    func compute(samples: [Float]) -> [Float]? {
        guard let stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor) else {
            return nil
        }
        defer { SherpaOnnxDestroyOnlineStream(stream) }
        samples.withUnsafeBufferPointer { ptr in
            SherpaOnnxOnlineStreamAcceptWaveform(stream, SherpaConstants.sampleRate, ptr.baseAddress, Int32(ptr.count))
        }
        SherpaOnnxOnlineStreamInputFinished(stream)
        guard SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, stream) == 1,
              let embeddingPtr = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream) else {
            return nil
        }
        let result = Array(UnsafeBufferPointer(start: embeddingPtr, count: Int(dim)))
        SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embeddingPtr)
        return result
    }

    deinit {
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor)
    }
}

// MARK: - Shared helpers

private enum SherpaConstants {
    static let sampleRate: Int32 = 16000
}

private enum SherpaAdapterError: LocalizedError {
    case missingComponent(String)
    case missingRole(String, String)
    case unsupportedEngine(String, String)
    case creationFailed(String)

    var errorDescription: String? {
        switch self {
        case let .missingComponent(name): return "模型清单缺少组件: \(name)"
        case let .missingRole(comp, role): return "组件 \(comp) 缺少角色: \(role)"
        case let .unsupportedEngine(comp, engine): return "组件 \(comp) 引擎不支持: \(engine)"
        case let .creationFailed(name): return "\(name) 创建失败"
        }
    }
}

private func makeOnlineModelConfig(
    packDir: URL, component: IOSModelPackManifest.Component,
    encoderRole: String, decoderRole: String, joinerRole: String, tokensRole: String
) -> SherpaOnnxOnlineModelConfig {
    var model = SherpaOnnxOnlineModelConfig()
    memset(&model, 0, MemoryLayout<SherpaOnnxOnlineModelConfig>.size)
    model.tokens = cstr(packDir.appendingPathComponent(tokensRole).path)
    model.num_threads = Int32(max(component.intOption("num_threads", fallback: 2), 1))
    model.provider = cstr("cpu")
    model.debug = 0
    switch component.engine {
    case "online_paraformer":
        model.paraformer.encoder = cstr(packDir.appendingPathComponent(encoderRole).path)
        model.paraformer.decoder = cstr(packDir.appendingPathComponent(decoderRole).path)
    case "online_transducer":
        model.transducer.encoder = cstr(packDir.appendingPathComponent(encoderRole).path)
        model.transducer.decoder = cstr(packDir.appendingPathComponent(decoderRole).path)
        model.transducer.joiner = cstr(packDir.appendingPathComponent(joinerRole).path)
    default:
        break
    }
    return model
}

private func freeOnlineModelConfig(_ model: inout SherpaOnnxOnlineModelConfig) {
    cstrFree(model.tokens)
    cstrFree(model.provider)
    cstrFree(model.model_type)
    cstrFree(model.modeling_unit)
    cstrFree(model.bpe_vocab)
    cstrFree(model.tokens_buf)
    cstrFree(model.transducer.encoder)
    cstrFree(model.transducer.decoder)
    cstrFree(model.transducer.joiner)
    cstrFree(model.paraformer.encoder)
    cstrFree(model.paraformer.decoder)
    cstrFree(model.zipformer2_ctc.model)
    cstrFree(model.nemo_ctc.model)
    cstrFree(model.t_one_ctc.model)
}

// MARK: - JSONOption helpers on manifest component

private extension IOSModelPackManifest.Component {
    func floatOption(_ key: String, fallback: Float) -> Float {
        guard let value = options[key] else { return fallback }
        if case let .number(n) = value { return Float(n) }
        return fallback
    }

    func intOption(_ key: String, fallback: Int) -> Int {
        guard let value = options[key] else { return fallback }
        if case let .number(n) = value { return Int(n) }
        return fallback
    }

    func stringOption(_ key: String, fallback: String) -> String {
        guard let value = options[key] else { return fallback }
        if case let .string(s) = value { return s }
        return fallback
    }

    func boolOption(_ key: String, fallback: Bool) -> Bool {
        guard let value = options[key] else { return fallback }
        if case let .bool(b) = value { return b }
        return fallback
    }
}
