import XCTest
@testable import AIGlassesMicProbe

final class IOSModelPackTests: XCTestCase {
    func testManifestRequiresAllFiveComponents() {
        let manifest = IOSModelPackManifest(schema: IOSModelPackManifest.schema, version: "2026.07.29", components: ["vad": component()])
        XCTAssertThrowsError(try manifest.validate())
    }

    func testManifestRejectsNonHTTPSModelFile() {
        let invalid = IOSModelPackManifest.File(name: "vad.onnx", url: "http://example.test/vad.onnx", sizeBytes: 8, sha256: String(repeating: "a", count: 64))
        let components = Dictionary(uniqueKeysWithValues: IOSModelPackManifest.requiredComponents.map { ($0, IOSModelPackManifest.Component(files: [invalid], options: [:])) })
        XCTAssertThrowsError(try IOSModelPackManifest(schema: IOSModelPackManifest.schema, version: "2026.07.29", components: components).validate())
    }

    private func component() -> IOSModelPackManifest.Component {
        IOSModelPackManifest.Component(files: [IOSModelPackManifest.File(name: "model.onnx", url: "https://example.test/model.onnx", sizeBytes: 8, sha256: String(repeating: "a", count: 64))], options: [:])
    }
}
