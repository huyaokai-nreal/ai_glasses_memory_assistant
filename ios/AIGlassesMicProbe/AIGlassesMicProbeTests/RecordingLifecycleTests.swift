import XCTest
@testable import AIGlassesMicProbe

final class RecordingLifecycleTests: XCTestCase {
    func testRouteVerificationTransitionsToRecording() {
        var lifecycle = RecordingLifecycle()

        XCTAssertFalse(lifecycle.apply(.startRequested))
        XCTAssertEqual(lifecycle.state, .preparing)
        XCTAssertFalse(lifecycle.apply(.routeVerified))
        XCTAssertEqual(lifecycle.state, .recording)
    }

    func testRouteChangeStopsAnActiveRecording() {
        var lifecycle = RecordingLifecycle()
        _ = lifecycle.apply(.startRequested)
        _ = lifecycle.apply(.routeVerified)

        XCTAssertTrue(lifecycle.apply(.routeChanged))
        XCTAssertEqual(lifecycle.state, .idle)
    }

    func testBackgroundEntryStopsAnActiveRecording() {
        var lifecycle = RecordingLifecycle()
        _ = lifecycle.apply(.startRequested)
        _ = lifecycle.apply(.routeVerified)

        XCTAssertTrue(lifecycle.apply(.enteredBackground))
        XCTAssertEqual(lifecycle.state, .idle)
    }

    func testFailureStopsAnActiveRecordingAndExposesReason() {
        var lifecycle = RecordingLifecycle()
        _ = lifecycle.apply(.startRequested)
        _ = lifecycle.apply(.routeVerified)

        XCTAssertTrue(lifecycle.apply(.failed("路由不匹配")))
        XCTAssertEqual(lifecycle.state, .failed("路由不匹配"))
    }
}
