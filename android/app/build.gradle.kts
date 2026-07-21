import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val repositoryRoot = rootProject.projectDir.parentFile
val sherpaVersion = "1.13.4"
val sherpaSha256 = "03f9c4df965f21c71269365a7951a7f23b5696fddd093fa318c80d65550ab780"
val sherpaAar = layout.buildDirectory.file("verified-dependencies/sherpa-onnx-$sherpaVersion.aar")
val fetchSherpaAar = tasks.register("fetchSherpaAar") {
    outputs.file(sherpaAar)
    doLast {
        val target = sherpaAar.get().asFile
        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        if (target.isFile && sha256(target) == sherpaSha256) return@doLast
        target.parentFile.mkdirs()
        val temporary = File(target.parentFile, ".${target.name}.part")
        temporary.delete()
        URI("https://github.com/k2-fsa/sherpa-onnx/releases/download/v$sherpaVersion/sherpa-onnx-$sherpaVersion.aar")
            .toURL()
            .openStream()
            .buffered()
            .use { input -> temporary.outputStream().buffered().use(input::copyTo) }
        val actualSha256 = sha256(temporary)
        check(actualSha256 == sherpaSha256) {
            temporary.delete()
            "sherpa-onnx AAR checksum mismatch: $actualSha256"
        }
        Files.move(
            temporary.toPath(),
            target.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }
}

android {
    namespace = "com.aiglasses.memoryassistant.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.aiglasses.memoryassistant.demo"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    sourceSets.getByName("main").assets.srcDir(repositoryRoot.resolve("static"))

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            install("numpy==1.26.2")
        }
    }
    sourceSets {
        getByName("main") {
            srcDir(repositoryRoot)
            include("ai_glasses_memory_assistant/**/*.py")
        }
    }
}

dependencies {
    implementation(files(fetchSherpaAar))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
