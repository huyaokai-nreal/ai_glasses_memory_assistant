import Foundation

enum RecordingState: Equatable {
    case idle
    case preparing
    case recording
    case failed(String)

    var isRecording: Bool {
        self == .recording
    }

    var isPreparing: Bool {
        self == .preparing
    }
}

enum RecordingEvent: Equatable {
    case startRequested
    case routeVerified
    case stopRequested
    case routeChanged
    case interrupted
    case enteredBackground
    case mediaServicesReset
    case failed(String)
}

struct RecordingLifecycle {
    private(set) var state: RecordingState = .idle

    @discardableResult
    mutating func apply(_ event: RecordingEvent) -> Bool {
        switch event {
        case .startRequested:
            guard !state.isRecording else { return false }
            state = .preparing
            return false
        case .routeVerified:
            guard state == .preparing else { return false }
            state = .recording
            return false
        case .stopRequested:
            let shouldStopRecorder = state.isRecording
            state = .idle
            return shouldStopRecorder
        case .routeChanged, .interrupted, .enteredBackground, .mediaServicesReset:
            let shouldStopRecorder = state.isRecording
            state = .idle
            return shouldStopRecorder
        case let .failed(message):
            let shouldStopRecorder = state.isRecording
            state = .failed(message)
            return shouldStopRecorder
        }
    }
}
