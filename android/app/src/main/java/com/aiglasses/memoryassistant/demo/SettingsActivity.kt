package com.aiglasses.memoryassistant.demo

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast

class SettingsActivity : Activity() {
    private lateinit var settings: SecureSettings
    private var pendingDiagnosticPassphrase = CharArray(0)
    private var diagnosticExportInProgress = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = SecureSettings(this)
        title = getString(R.string.settings)

        val provider = field("Provider", settings.provider())
        val model = field("Model", settings.model())
        val baseUrl = field("Base URL", settings.baseUrl()).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        }
        val apiKey = field("DeepSeek API key", settings.apiKey()).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val modelManifestUrl = field("模型清单 URL（HTTPS）", settings.modelManifestUrl()).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        }
        val installedVersion = ModelPackInstaller(this).currentVersion()
        ModelPackState.markExisting(installedVersion)
        val restoredSelfTest = ModelSelfTestState.restore(this, installedVersion)
        val modelStatus = TextView(this).apply {
            text = modelStatusText(installedVersion, restoredSelfTest)
            setPadding(0, dp(12), 0, dp(12))
        }
        val owner = TextView(this).apply {
            text = getString(R.string.owner_id_label, settings.ownerId())
            setPadding(0, dp(12), 0, dp(12))
        }
        val save = Button(this).apply {
            text = "保存并返回"
            setOnClickListener {
                runCatching {
                    settings.save(
                        provider = provider.text.toString(),
                        model = model.text.toString(),
                        baseUrl = baseUrl.text.toString(),
                        apiKey = apiKey.text.toString(),
                        modelManifestUrl = modelManifestUrl.text.toString(),
                    )
                }.onSuccess {
                    Toast.makeText(this@SettingsActivity, "配置已保存", Toast.LENGTH_SHORT).show()
                    finish()
                }.onFailure { error ->
                    Toast.makeText(this@SettingsActivity, error.message ?: "配置无效", Toast.LENGTH_LONG).show()
                }
            }
        }
        val installModels = Button(this).apply {
            text = "下载或更新本地模型"
            isEnabled = !NativeAudioState.snapshot().running
            setOnClickListener {
                runCatching {
                    val url = modelManifestUrl.text.toString().trim()
                    require(url.isNotEmpty()) { "请先填写模型清单 URL" }
                    settings.save(
                        provider = provider.text.toString(),
                        model = model.text.toString(),
                        baseUrl = baseUrl.text.toString(),
                        apiKey = apiKey.text.toString(),
                        modelManifestUrl = url,
                    )
                    ModelDownloadService.start(this@SettingsActivity, url)
                }.onSuccess {
                    Toast.makeText(this@SettingsActivity, "模型下载已开始", Toast.LENGTH_SHORT).show()
                    finish()
                }.onFailure { error ->
                    Toast.makeText(this@SettingsActivity, error.message ?: "无法开始模型下载", Toast.LENGTH_LONG).show()
                }
            }
        }
        val selfTestModels = Button(this).apply {
            text = "运行模型自检"
            isEnabled = installedVersion != null && !NativeAudioState.snapshot().running
        }
        selfTestModels.setOnClickListener {
            selfTestModels.isEnabled = false
            modelStatus.text = "本地模型：正在逐项自检…"
            Thread({
                val result = runCatching {
                    val pack = ModelPackInstaller(this@SettingsActivity).current()
                        ?: error("未找到完整且校验通过的模型包")
                    ModelSelfTestRunner(this@SettingsActivity).run(pack)
                }.getOrElse { error ->
                    ModelSelfTestSnapshot(
                        state = "failed",
                        version = installedVersion.orEmpty(),
                        checkedAtMillis = System.currentTimeMillis(),
                        components = errorComponent(error),
                    ).also { ModelSelfTestState.save(this@SettingsActivity, it) }
                }
                runCatching { PythonRuntime.setDeviceState(org.json.JSONObject(NativeAudioState.snapshotJson())) }
                runOnUiThread {
                    modelStatus.text = modelStatusText(installedVersion, result)
                    selfTestModels.isEnabled = true
                    Toast.makeText(
                        this@SettingsActivity,
                        if (result.state == "ok") "五项模型自检通过" else "模型自检未通过，请查看逐项错误",
                        Toast.LENGTH_LONG,
                    ).show()
                }
            }, "model-self-test").start()
        }
        val exportDiagnostics = Button(this).apply {
            text = "导出加密诊断包"
            isEnabled = !NativeAudioState.snapshot().running
            setOnClickListener { requestDiagnosticExport() }
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(24))
            addView(owner)
            addLabeledField("Provider", provider)
            addLabeledField("Model", model)
            addLabeledField("Base URL", baseUrl)
            addLabeledField("DeepSeek API key", apiKey)
            addLabeledField("模型清单 URL", modelManifestUrl)
            addView(modelStatus)
            addView(selfTestModels, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(installModels, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(exportDiagnostics, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(save, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        setContentView(ScrollView(this).apply { addView(content) })
    }

    override fun onDestroy() {
        pendingDiagnosticPassphrase.fill('\u0000')
        pendingDiagnosticPassphrase = CharArray(0)
        super.onDestroy()
    }

    @Deprecated("Activity result API is sufficient for this framework-only demo")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_DIAGNOSTIC_EXPORT) return
        val passphrase = pendingDiagnosticPassphrase
        pendingDiagnosticPassphrase = CharArray(0)
        val destination = data?.data
        if (resultCode != RESULT_OK || destination == null) {
            passphrase.fill('\u0000')
            return
        }
        if (diagnosticExportInProgress) {
            passphrase.fill('\u0000')
            return
        }
        diagnosticExportInProgress = true
        Thread({
            val result = runCatching { DiagnosticExporter(this).export(destination, passphrase) }
            runOnUiThread {
                diagnosticExportInProgress = false
                result.onSuccess {
                    Toast.makeText(this, "加密诊断包已导出", Toast.LENGTH_LONG).show()
                }.onFailure { error ->
                    Toast.makeText(this, error.message ?: "诊断包导出失败", Toast.LENGTH_LONG).show()
                }
            }
        }, "diagnostic-export").start()
    }

    private fun field(label: String, value: String) = EditText(this).apply {
        hint = label
        setText(value)
        setSingleLine(true)
    }

    private fun LinearLayout.addLabeledField(label: String, field: EditText) {
        addView(TextView(this@SettingsActivity).apply { text = label })
        addView(field, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun modelStatusText(version: String?, selfTest: ModelSelfTestSnapshot): String {
        if (version == null) return "本地模型：未安装"
        val labels = mapOf(
            "vad" to "VAD",
            "kws" to "KWS",
            "online_asr" to "在线 ASR",
            "ambient_asr" to "SenseVoice",
            "speaker" to "声纹",
        )
        val lines = mutableListOf("本地模型：$version", "自检：${selfTest.state}")
        labels.forEach { (name, label) ->
            val result = selfTest.components[name] ?: return@forEach
            val detail = if (result.error.isBlank()) "" else " · ${result.error}"
            lines += "$label：${result.state} · ${result.elapsedMs} ms$detail"
        }
        if (selfTest.pssKb > 0) lines += "App PSS：${selfTest.pssKb / 1024} MB"
        return lines.joinToString("\n")
    }

    private fun errorComponent(error: Throwable): Map<String, ModelComponentSelfTest> = mapOf(
        "pack" to ModelComponentSelfTest(
            state = "failed",
            elapsedMs = 0,
            error = listOf(error.javaClass.simpleName, error.message.orEmpty()).filter(String::isNotBlank).joinToString(": ").take(300),
        ),
    )

    private fun requestDiagnosticExport() {
        if (NativeAudioState.snapshot().running) {
            Toast.makeText(this, "请先停止全天收音", Toast.LENGTH_LONG).show()
            return
        }
        val first = passwordField("导出密码（至少 8 个字符）")
        val confirmation = passwordField("再次输入导出密码")
        val fields = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), 0)
            addView(first)
            addView(confirmation)
        }
        AlertDialog.Builder(this)
            .setTitle("加密诊断包")
            .setView(fields)
            .setNegativeButton("取消", null)
            .setPositiveButton("选择保存位置") { _, _ ->
                val passphrase = first.text.toString().toCharArray()
                val repeated = confirmation.text.toString().toCharArray()
                first.text.clear()
                confirmation.text.clear()
                if (passphrase.size < 8 || !passphrase.contentEquals(repeated)) {
                    passphrase.fill('\u0000')
                    repeated.fill('\u0000')
                    Toast.makeText(this, "密码至少 8 个字符且两次输入必须一致", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                repeated.fill('\u0000')
                pendingDiagnosticPassphrase.fill('\u0000')
                pendingDiagnosticPassphrase = passphrase
                val fileName = "ai-glasses-diagnostic-${java.time.LocalDateTime.now().format(DIAGNOSTIC_TIME)}.aigd"
                startActivityForResult(
                    Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "application/octet-stream"
                        putExtra(Intent.EXTRA_TITLE, fileName)
                    },
                    REQUEST_DIAGNOSTIC_EXPORT,
                )
            }
            .show()
    }

    private fun passwordField(label: String) = EditText(this).apply {
        hint = label
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        setSingleLine(true)
    }

    companion object {
        private const val REQUEST_DIAGNOSTIC_EXPORT = 210
        private val DIAGNOSTIC_TIME = java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
    }
}
