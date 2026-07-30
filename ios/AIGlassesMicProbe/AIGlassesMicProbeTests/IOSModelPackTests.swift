import XCTest
@testable import AIGlassesMicProbe

final class IOSModelPackTests: XCTestCase {
    func testManifestRequiresAllFiveComponents() {
        let manifest = manifest(components: ["vad": component()])
        XCTAssertThrowsError(try manifest.validate())
    }

    func testManifestRejectsRoleWithoutDeclaredFile() {
        let components = Dictionary(uniqueKeysWithValues: IOSModelPackManifest.requiredComponents.map { ($0, component(role: "missing/model.onnx")) })
        XCTAssertThrowsError(try manifest(components: components).validate())
    }

    func testCompleteAndroidCompatibleManifestPasses() throws {
        let components = Dictionary(uniqueKeysWithValues: IOSModelPackManifest.requiredComponents.map { ($0, component()) })
        XCTAssertNoThrow(try manifest(components: components).validate())
    }

    private func manifest(components: [String: IOSModelPackManifest.Component]) -> IOSModelPackManifest {
        IOSModelPackManifest(
            schema: IOSModelPackManifest.schema,
            version: "test-1",
            sherpaOnnxVersion: IOSModelPackManifest.sherpaOnnxVersion,
            installMode: "adb_local",
            files: [IOSModelPackManifest.File(path: "shared/model.onnx", sizeBytes: 8, sha256: String(repeating: "a", count: 64))],
            components: components
        )
    }

    private func component(role: String = "shared/model.onnx") -> IOSModelPackManifest.Component {
        IOSModelPackManifest.Component(engine: "test", roles: ["model": role], options: [:])
    }
}
