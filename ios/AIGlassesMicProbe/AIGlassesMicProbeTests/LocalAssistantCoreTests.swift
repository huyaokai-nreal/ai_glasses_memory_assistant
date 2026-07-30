import XCTest
@testable import AIGlassesMicProbe

final class LocalAssistantCoreTests: XCTestCase {
    func testPartialNeverPersists() throws {
        let store = try LocalAssistantStore(directory: temporaryDirectory())

        let result = try store.ingest(LocalAudioEvent(kind: .partial, transcript: "debug partial", speakerState: .owner, overlapState: .none, privacyApproved: true))

        XCTAssertFalse(result.accepted)
        XCTAssertEqual(result.reason, "partial_ui_only")
        XCTAssertEqual(try store.snapshot().timeline, [])
        XCTAssertEqual(try store.snapshot().memories, [])
        XCTAssertEqual(try store.snapshot().audit, [])
    }

    func testOnlyApprovedOwnerFinalCanWriteMemory() throws {
        let store = try LocalAssistantStore(directory: temporaryDirectory())
        let rejected = try store.ingest(LocalAudioEvent(kind: .final, transcript: "请记住这是他人说的话", speakerState: .other, overlapState: .none, privacyApproved: true))
        let accepted = try store.ingest(LocalAudioEvent(kind: .final, transcript: "请记住我喜欢黑咖啡", speakerState: .owner, overlapState: .none, privacyApproved: true))

        XCTAssertFalse(rejected.accepted)
        XCTAssertEqual(rejected.reason, "speaker_other")
        XCTAssertTrue(accepted.accepted)
        XCTAssertEqual(accepted.memoryCount, 1)
        XCTAssertEqual(try store.snapshot().timeline.count, 1)
    }

    func testDiagnosticUsesAIGDIAG1AndDoesNotContainApiKey() throws {
        let snapshot = LocalAssistantSnapshot(schema: "ios_local_assistant_poc.v1", generatedAt: Date(timeIntervalSince1970: 0), timeline: [], memories: [], audit: [])
        let encrypted = try DiagnosticBundleExporter.encrypt(snapshot: snapshot, passphrase: "password-123")

        XCTAssertEqual(encrypted.prefix(8), Data("AIGDIAG1".utf8))
        XCTAssertFalse(encrypted.contains(Data("api-key".utf8)))
    }

    private func temporaryDirectory() -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }
}
