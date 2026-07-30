import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

enum LocalAssistantError: LocalizedError {
    case database(String)

    var errorDescription: String? {
        switch self {
        case let .database(message): return "本机数据库错误：\(message)"
        }
    }
}

struct LocalAudioEvent: Equatable {
    enum Kind: String { case partial, final }
    enum SpeakerState: String { case owner, other, unknown, environment }
    enum OverlapState: String { case none, suspected, unknown }

    let id: String
    let kind: Kind
    let transcript: String
    let speakerState: SpeakerState
    let overlapState: OverlapState
    let privacyApproved: Bool
    let createdAt: Date

    init(
        id: String = UUID().uuidString,
        kind: Kind,
        transcript: String,
        speakerState: SpeakerState,
        overlapState: OverlapState,
        privacyApproved: Bool,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.kind = kind
        self.transcript = transcript
        self.speakerState = speakerState
        self.overlapState = overlapState
        self.privacyApproved = privacyApproved
        self.createdAt = createdAt
    }
}

struct LocalPersistenceResult: Equatable {
    let accepted: Bool
    let reason: String
    let timelineCount: Int
    let memoryCount: Int
}

struct LocalAssistantSnapshot: Codable, Equatable {
    let schema: String
    let generatedAt: Date
    let timeline: [Timeline]
    let memories: [Memory]
    let audit: [Audit]

    struct Timeline: Codable, Equatable {
        let eventID: String
        let text: String
        let createdAt: Date
    }

    struct Memory: Codable, Equatable {
        let id: String
        let content: String
        let sourceEventID: String
        let createdAt: Date
    }

    struct Audit: Codable, Equatable {
        let eventID: String
        let action: String
        let reason: String
        let createdAt: Date
    }
}

/// Native fallback POC storage. It deliberately accepts only a privacy-approved owner final.
final class LocalAssistantStore {
    private var database: OpaquePointer?
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let databaseURL = directory.appendingPathComponent("memory-assistant-poc.sqlite")
        guard sqlite3_open_v2(databaseURL.path, &database, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
            throw LocalAssistantError.database("无法打开 SQLite")
        }
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
        try execute("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS timeline (
            event_id TEXT PRIMARY KEY NOT NULL,
            text TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY NOT NULL,
            content TEXT NOT NULL,
            source_event_id TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """)
    }

    deinit { sqlite3_close(database) }

    func ingest(_ event: LocalAudioEvent) throws -> LocalPersistenceResult {
        let gate = gate(event)
        guard gate.allowed else {
            // A rejected final is auditable by its opaque event ID and reason, never its transcript.
            if event.kind == .final { try insertAudit(eventID: event.id, action: "rejected_final", reason: gate.reason, createdAt: event.createdAt) }
            return try result(accepted: false, reason: gate.reason)
        }

        try insertTimeline(event)
        let memoryContent = memoryCandidate(from: event.transcript)
        if let memoryContent {
            try insertMemory(content: memoryContent, sourceEventID: event.id, createdAt: event.createdAt)
            try insertAudit(eventID: event.id, action: "memory_saved", reason: "explicit_memory_request", createdAt: event.createdAt)
        } else {
            try insertAudit(eventID: event.id, action: "timeline_saved", reason: "final_without_memory_request", createdAt: event.createdAt)
        }
        return try result(accepted: true, reason: memoryContent == nil ? "timeline_saved" : "memory_saved")
    }

    func snapshot() throws -> LocalAssistantSnapshot {
        LocalAssistantSnapshot(
            schema: "ios_local_assistant_poc.v1",
            generatedAt: Date(),
            timeline: try rows("SELECT event_id, text, created_at FROM timeline ORDER BY created_at") { statement in
                LocalAssistantSnapshot.Timeline(
                    eventID: Self.string(statement, 0), text: Self.string(statement, 1), createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 2))
                )
            },
            memories: try rows("SELECT id, content, source_event_id, created_at FROM memories ORDER BY created_at") { statement in
                LocalAssistantSnapshot.Memory(
                    id: Self.string(statement, 0), content: Self.string(statement, 1), sourceEventID: Self.string(statement, 2), createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 3))
                )
            },
            audit: try rows("SELECT event_id, action, reason, created_at FROM audit ORDER BY id") { statement in
                LocalAssistantSnapshot.Audit(
                    eventID: Self.string(statement, 0), action: Self.string(statement, 1), reason: Self.string(statement, 2), createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 3))
                )
            }
        )
    }

    private func gate(_ event: LocalAudioEvent) -> (allowed: Bool, reason: String) {
        guard event.kind == .final else { return (false, "partial_ui_only") }
        guard !event.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return (false, "empty_final") }
        guard event.speakerState == .owner else { return (false, "speaker_\(event.speakerState.rawValue)") }
        guard event.overlapState == .none else { return (false, "overlap_\(event.overlapState.rawValue)") }
        guard event.privacyApproved else { return (false, "privacy_not_approved") }
        guard !containsSensitiveData(event.transcript) else { return (false, "sensitive_content") }
        return (true, "allowed")
    }

    private func memoryCandidate(from transcript: String) -> String? {
        let markers = ["请记住", "记住", "记一下", "保存"]
        guard let marker = markers.first(where: { transcript.contains($0) }) else { return nil }
        let content = transcript.replacingOccurrences(of: marker, with: "").trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        return content.isEmpty ? nil : content
    }

    private func containsSensitiveData(_ text: String) -> Bool {
        let lowered = text.lowercased()
        let terms = ["密码", "验证码", "银行卡", "身份证", "护照", "api key", "apikey", "token", "secret", "bearer", "jwt"]
        return terms.contains { lowered.contains($0) }
    }

    private func insertTimeline(_ event: LocalAudioEvent) throws {
        try bind("INSERT INTO timeline(event_id, text, created_at) VALUES (?, ?, ?)", values: [.text(event.id), .text(event.transcript), .number(event.createdAt.timeIntervalSince1970)])
    }

    private func insertMemory(content: String, sourceEventID: String, createdAt: Date) throws {
        try bind("INSERT INTO memories(id, content, source_event_id, created_at) VALUES (?, ?, ?, ?)", values: [.text(UUID().uuidString), .text(content), .text(sourceEventID), .number(createdAt.timeIntervalSince1970)])
    }

    private func insertAudit(eventID: String, action: String, reason: String, createdAt: Date) throws {
        try bind("INSERT INTO audit(event_id, action, reason, created_at) VALUES (?, ?, ?, ?)", values: [.text(eventID), .text(action), .text(reason), .number(createdAt.timeIntervalSince1970)])
    }

    private func result(accepted: Bool, reason: String) throws -> LocalPersistenceResult {
        let snapshot = try snapshot()
        return LocalPersistenceResult(accepted: accepted, reason: reason, timelineCount: snapshot.timeline.count, memoryCount: snapshot.memories.count)
    }

    private enum BoundValue { case text(String), number(Double) }

    private func execute(_ sql: String) throws {
        var error: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(database, sql, nil, nil, &error) == SQLITE_OK else {
            defer { sqlite3_free(error) }
            throw LocalAssistantError.database(error.map { String(cString: $0) } ?? "SQL 执行失败")
        }
    }

    private func bind(_ sql: String, values: [BoundValue]) throws {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK, let statement else { throw failure() }
        defer { sqlite3_finalize(statement) }
        for (offset, value) in values.enumerated() {
            let index = Int32(offset + 1)
            let status: Int32
            switch value {
            case let .text(value): status = sqlite3_bind_text(statement, index, value, -1, SQLITE_TRANSIENT)
            case let .number(value): status = sqlite3_bind_double(statement, index, value)
            }
            guard status == SQLITE_OK else { throw failure() }
        }
        guard sqlite3_step(statement) == SQLITE_DONE else { throw failure() }
    }

    private func rows<Row>(_ sql: String, transform: (OpaquePointer) -> Row) throws -> [Row] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK, let statement else { throw failure() }
        defer { sqlite3_finalize(statement) }
        var output: [Row] = []
        while sqlite3_step(statement) == SQLITE_ROW { output.append(transform(statement)) }
        return output
    }

    private func failure() -> LocalAssistantError {
        LocalAssistantError.database(database.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "未知 SQLite 错误")
    }

    private static func string(_ statement: OpaquePointer, _ column: Int32) -> String {
        sqlite3_column_text(statement, column).map { String(cString: $0) } ?? ""
    }
}
