import XCTest
@testable import AIGlassesMicProbe

final class AudioRoutePolicyTests: XCTestCase {
    func testRejectsWhenNoBluetoothHFPInputExists() {
        let inputs = [AudioInputRoute(id: "built-in", name: "iPhone", kind: .builtInMic)]

        XCTAssertThrowsError(try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: inputs)) { error in
            XCTAssertEqual(error as? AudioRouteError, .noBluetoothMicrophone)
        }
    }

    func testSelectsTheOnlyBluetoothHFPInput() throws {
        let mic = AudioInputRoute(id: "mic-pro", name: "Insta Mic Pro", kind: .bluetoothHFP)
        let selected = try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: [mic])

        XCTAssertEqual(selected, mic)
    }

    func testRejectsMultipleBluetoothHFPInputs() {
        let first = AudioInputRoute(id: "one", name: "Mic Pro", kind: .bluetoothHFP)
        let second = AudioInputRoute(id: "two", name: "Headset", kind: .bluetoothHFP)

        XCTAssertThrowsError(try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: [first, second])) { error in
            XCTAssertEqual(error as? AudioRouteError, .ambiguousBluetoothMicrophones([first, second]))
        }
    }

    func testRejectsA2DPOnlyBluetoothRoute() {
        let outputOnly = AudioInputRoute(id: "speaker", name: "Bluetooth Speaker", kind: .bluetoothA2DP)

        XCTAssertThrowsError(try AudioRoutePolicy.selectPreferredBluetoothMicrophone(from: [outputOnly])) { error in
            XCTAssertEqual(error as? AudioRouteError, .noBluetoothMicrophone)
        }
    }

    func testRejectsActualRouteThatDoesNotMatchPreferredMicrophone() {
        let expected = AudioInputRoute(id: "mic-pro", name: "Insta Mic Pro", kind: .bluetoothHFP)
        let actual = AudioInputRoute(id: "built-in", name: "iPhone", kind: .builtInMic)

        XCTAssertThrowsError(try AudioRoutePolicy.verifyActiveInput(expected: expected, actual: actual)) { error in
            XCTAssertEqual(error as? AudioRouteError, .routeMismatch(expected: expected, actual: actual))
        }
    }

    func testRejectsPreferredMicrophoneThatDisappearsBeforeActivation() {
        let mic = AudioInputRoute(id: "mic-pro", name: "Insta Mic Pro", kind: .bluetoothHFP)

        XCTAssertThrowsError(try AudioRoutePolicy.requirePreferredInputAvailable(mic, in: [])) { error in
            XCTAssertEqual(error as? AudioRouteError, .preferredInputUnavailable("Insta Mic Pro"))
        }
    }
}
