import AVFoundation
import Foundation

// MARK: - Data models (mirror Android's OfflineAudioTestResult)

struct OfflineTestSegment {
    let startMillis: Int
    let endMillis: Int
    let text: String
}

struct OfflineTestResult {
    let modelVersion: String
    let sourceFormat: String
    let sourceSampleRate: Double
    let sourceChannelCount: Int
    let sourceDurationSec: Double
    let normalizedSampleCount: Int
    let gainEnabled: Bool
    let gainDecibels: Int
    let elapsedMillis: UInt64
    let segments: [OfflineTestSegment]
    var transcript: String { segments.map(\.text).joined(separator: "\n") }
}

// MARK: - Offline Audio Tester

/// Decodes WAV/M4A, resamples to 16 kHz mono, then runs VAD + SenseVoice ASR.
/// Mirrors Android's OfflineAudioTestRunner.
final class OfflineAudioTester {
    private let packDir: URL
    private let manifest: IOSModelPackManifest
    private var cancelled = false

    init(packDir: URL, manifest: IOSModelPackManifest) {
        self.packDir = packDir
        self.manifest = manifest
    }

    func run(fileURL: URL, gainEnabled: Bool, gainDecibels: Int) throws -> OfflineTestResult {
        cancelled = false
        let startTime = Date()

        // 1) Decode audio file
        let file = try AVAudioFile(forReading: fileURL)
        let originalFormat = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: originalFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw TesterError.decodeFailed("无法创建解码缓冲区")
        }
        try file.read(into: buffer)
        guard let channelData = buffer.floatChannelData else {
            throw TesterError.decodeFailed("无法读取音频数据")
        }
        let channelCount = Int(originalFormat.channelCount)
        let totalFrames = Int(buffer.frameLength)

        // 2) Downmix to mono (average channels)
        var mono: [Float]
        if channelCount == 1 {
            mono = Array(UnsafeBufferPointer(start: channelData[0], count: totalFrames))
        } else {
            mono = [Float](repeating: 0, count: totalFrames)
            for ch in 0..<channelCount {
                let ptr = channelData[ch]
                for i in 0..<totalFrames { mono[i] += ptr[i] }
            }
            let chFloat = Float(channelCount)
            for i in 0..<totalFrames { mono[i] /= chFloat }
        }

        // 3) Resample to 16 kHz (linear interpolation for simplicity)
        let targetRate: Double = 16_000
        let sourceRate = originalFormat.sampleRate
        var resampled: [Float]
        if abs(sourceRate - targetRate) < 1 {
            resampled = mono
        } else {
            let ratio = targetRate / sourceRate
            let outCount = Int(Double(totalFrames) * ratio)
            resampled = [Float](repeating: 0, count: outCount)
            for i in 0..<outCount {
                let srcIndex = Double(i) / ratio
                let srcLow = Int(srcIndex)
                let srcHigh = min(srcLow + 1, totalFrames - 1)
                let frac = Float(srcIndex - Double(srcLow))
                resampled[i] = mono[srcLow] * (1 - frac) + mono[srcHigh] * frac
            }
        }

        // 4) Apply gain
        var gain = OfflineTestGain(decibels: 0, enabled: false)
        if gainEnabled && gainDecibels != 0 {
            let multiplier = powf(10, Float(gainDecibels) / 20)
            for i in 0..<resampled.count { resampled[i] *= multiplier }
            gain = OfflineTestGain(decibels: gainDecibels, enabled: true)
        }

        // 5) Clip to [-1, 1]
        for i in 0..<resampled.count {
            resampled[i] = max(-1, min(1, resampled[i]))
        }

        // 6) Run VAD + ASR
        // SherpaAmbientAsrAdapter is a stateless offline recognizer: every
        // recognize() call is a self-contained SenseVoice inference on the
        // provided samples, so there's no internal buffer to flush.
        let vad = try SherpaVadAdapter(packDir: packDir, manifest: manifest)
        let asr = try SherpaAmbientAsrAdapter(packDir: packDir, manifest: manifest)

        var segments: [OfflineTestSegment] = []
        let frameSamples = 4000 // 250 ms at 16 kHz
        var offset = 0
        while offset < resampled.count && !cancelled {
            let end = min(offset + frameSamples, resampled.count)
            let chunk = Array(resampled[offset..<end])

            let vadSegments = vad.accept(samples: chunk)
            if !vadSegments.isEmpty {
                // Process accumulated VAD segments through offline ASR
                for vadSeg in vadSegments {
                    let startMs = (offset - chunk.count + vadSeg.start) * 1000 / Int(targetRate)
                    let endMs = startMs + vadSeg.samples.count * 1000 / Int(targetRate)
                    let text = asr.recognize(samples: vadSeg.samples).text
                    segments.append(OfflineTestSegment(startMillis: startMs, endMillis: endMs, text: text))
                }
            }
            offset = end
        }

        // Flush any tail speech the VAD is still holding (silence detected)
        let remainingVad = vad.flush()
        for vadSeg in remainingVad {
            let startMs = (resampled.count - vadSeg.samples.count) * 1000 / Int(targetRate)
            let endMs = resampled.count * 1000 / Int(targetRate)
            let text = asr.recognize(samples: vadSeg.samples).text
            segments.append(OfflineTestSegment(startMillis: startMs, endMillis: endMs, text: text))
        }

        let elapsed = UInt64(Date().timeIntervalSince(startTime) * 1000)

        return OfflineTestResult(
            modelVersion: manifest.version,
            sourceFormat: fileURL.pathExtension.uppercased(),
            sourceSampleRate: originalFormat.sampleRate,
            sourceChannelCount: channelCount,
            sourceDurationSec: Double(totalFrames) / originalFormat.sampleRate,
            normalizedSampleCount: resampled.count,
            gainEnabled: gain.enabled,
            gainDecibels: gain.decibels,
            elapsedMillis: elapsed,
            segments: segments
        )
    }

    func cancel() { cancelled = true }
}

struct OfflineTestGain {
    let decibels: Int
    let enabled: Bool
}

enum TesterError: LocalizedError {
    case decodeFailed(String)
    var errorDescription: String? {
        switch self {
        case let .decodeFailed(msg): return "音频解码失败：\(msg)"
        }
    }
}
