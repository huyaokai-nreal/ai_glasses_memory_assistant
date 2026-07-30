import Foundation
import SherpaOnnxC

enum SherpaRuntimeProbe {
    static func version() -> String {
        guard let version = SherpaOnnxGetVersionStr() else { return "unknown" }
        return String(cString: version)
    }
}
