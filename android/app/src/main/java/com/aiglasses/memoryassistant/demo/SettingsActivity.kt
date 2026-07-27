package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.util.Locale

class SettingsActivity : Activity() {
    private lateinit var settings: SecureSettings
    private var pendingDiagnosticPassphrase = CharArray(0)
    private var diagnosticExportInProgress = false
    private var audioInputProbe: AudioInputProbe? = null
    private var audioInputRecording: AudioInputProbeRecording? = null
    private var pendingAudioInputProbe = false
    private lateinit var audioInputProbeStatus: TextView
    private lateinit var audioInputProbeButton: Button
    private lateinit var audioInputPlaybackButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = SecureSettings(this)
        title = getString(R.string.advanced_settings)

        val provider = field("服务提供商", settings.provider())
        val model = field("模型名称", settings.model())
        val baseUrl = field("API 地址", settings.baseUrl()).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        }
        val apiKey = field("API 密钥", settings.apiKey()).apply {
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
            styleActionButton(primary = true)
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
            styleActionButton()
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
            styleActionButton()
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
            styleActionButton()
            isEnabled = !NativeAudioState.snapshot().running
            setOnClickListener { requestDiagnosticExport() }
        }
        audioInputProbeStatus = TextView(this).apply {
            text = "测试前请停止全天收音；蓝牙设备接入时会优先且仅使用蓝牙收音"
            setPadding(0, dp(12), 0, dp(12))
        }
        audioInputProbeButton = Button(this).apply {
            text = "测试收音"
            styleActionButton()
            setOnClickListener { requestAudioInputProbe() }
        }
        audioInputPlaybackButton = Button(this).apply {
            text = "播放本次测试录音"
            styleActionButton()
            visibility = View.GONE
            setOnClickListener { playAudioInputRecording() }
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(24))
            setBackgroundColor(Color.rgb(246, 247, 249))
            addView(sectionTitle("账户与服务"))
            addView(owner)
            addLabeledField("服务提供商", provider)
            addLabeledField("模型名称", model)
            addLabeledField("API 地址", baseUrl)
            addLabeledField("API 密钥", apiKey)
            addView(sectionTitle("本地语音模型"))
            addLabeledField("模型清单 URL", modelManifestUrl)
            addView(modelStatus)
            addActionButton(selfTestModels)
            addActionButton(installModels)
            addView(sectionTitle("收音设备"))
            addView(audioInputProbeStatus)
            addActionButton(audioInputProbeButton)
            addActionButton(audioInputPlaybackButton)
            addView(sectionTitle("诊断与保存"))
            addActionButton(exportDiagnostics)
            addActionButton(save)
        }
        val toolbar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(8), dp(16), dp(8))
            setBackgroundColor(Color.WHITE)
            elevation = dp(2).toFloat()
            addView(ImageButton(this@SettingsActivity).apply {
                contentDescription = getString(R.string.back)
                setImageResource(R.drawable.ic_arrow_back)
                background = null
                setPadding(dp(12), dp(12), dp(12), dp(12))
                setOnClickListener { finish() }
            }, ViewGroup.LayoutParams(dp(48), dp(48)))
            addView(LinearLayout(this@SettingsActivity).apply {
                orientation = LinearLayout.VERTICAL
                addView(TextView(this@SettingsActivity).apply {
                    text = getString(R.string.advanced_settings)
                    textSize = 20f
                    setTextColor(Color.rgb(32, 33, 35))
                    setTypeface(typeface, Typeface.BOLD)
                })
                addView(TextView(this@SettingsActivity).apply {
                    text = "模型、API 与本机诊断"
                    textSize = 12f
                    setTextColor(Color.rgb(107, 114, 128))
                })
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(content)
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(246, 247, 249))
            addView(toolbar, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        })
    }

    override fun onDestroy() {
        audioInputProbe?.cancel()
        audioInputProbe = null
        clearAudioInputRecording()
        pendingDiagnosticPassphrase.fill('\u0000')
        pendingDiagnosticPassphrase = CharArray(0)
        super.onDestroy()
    }

    override fun onStop() {
        super.onStop()
        if (audioInputProbe != null) {
            audioInputProbe?.cancel()
            audioInputProbe = null
            audioInputProbeButton.isEnabled = true
        }
        clearAudioInputRecording()
        audioInputProbeStatus.text = "测试前请停止全天收音；蓝牙设备接入时会优先且仅使用蓝牙收音"
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_AUDIO_INPUT_PROBE) return
        val granted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (granted && pendingAudioInputProbe) {
            pendingAudioInputProbe = false
            startAudioInputProbe()
        } else if (!granted) {
            pendingAudioInputProbe = false
            audioInputProbeStatus.text = "测试失败：未授予麦克风权限"
            Toast.makeText(this, "需要麦克风权限才能测试收音", Toast.LENGTH_LONG).show()
        }
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
        addView(TextView(this@SettingsActivity).apply {
            text = label
            textSize = 13f
            setTextColor(Color.rgb(75, 85, 99))
            setPadding(0, dp(8), 0, dp(4))
        })
        addView(field, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            bottomMargin = dp(6)
        })
    }

    private fun LinearLayout.addActionButton(button: Button) {
        addView(button, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            topMargin = dp(8)
        })
    }

    private fun sectionTitle(text: String) = TextView(this).apply {
        this.text = text
        textSize = 14f
        setTextColor(Color.rgb(15, 118, 110))
        setTypeface(typeface, Typeface.BOLD)
        setPadding(0, dp(16), 0, dp(6))
    }

    private fun Button.styleActionButton(primary: Boolean = false) {
        minHeight = dp(48)
        isAllCaps = false
        backgroundTintList = ColorStateList.valueOf(
            if (primary) Color.rgb(15, 118, 110) else Color.rgb(31, 41, 55),
        )
        setTextColor(Color.WHITE)
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

    private fun requestAudioInputProbe() {
        if (NativeAudioState.snapshot().running) {
            audioInputProbeStatus.text = "请先停止全天收音，再测试蓝牙收音设备"
            Toast.makeText(this, "请先停止全天收音", Toast.LENGTH_LONG).show()
            return
        }
        if (audioInputProbe != null) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingAudioInputProbe = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_AUDIO_INPUT_PROBE)
            return
        }
        startAudioInputProbe()
    }

    private fun startAudioInputProbe() {
        if (NativeAudioState.snapshot().running || audioInputProbe != null) return
        clearAudioInputRecording()
        audioInputProbeButton.isEnabled = false
        audioInputProbeStatus.text = "正在测试收音，请对蓝牙收音设备说话…"
        val probe = AudioInputProbe(applicationContext) { update ->
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                renderAudioInputProbe(update)
                if (!update.testing) {
                    update.recording?.let { recording ->
                        audioInputRecording = recording
                        audioInputPlaybackButton.visibility = View.VISIBLE
                    }
                    audioInputProbe = null
                    audioInputProbeButton.isEnabled = true
                }
            }
        }
        audioInputProbe = probe
        probe.start()
    }

    private fun renderAudioInputProbe(update: AudioInputProbeUpdate) {
        val route = update.snapshot.route
        val lines = mutableListOf<String>()
        lines += "当前收音设备：${route?.label ?: "未确认"}"
        route?.let {
            lines += "设备类型：${it.typeLabel}"
            lines += "蓝牙输入：${if (it.isBluetooth) "是" else "否"}"
        }
        lines += "当前峰值：${String.format(Locale.US, "%.1f", update.snapshot.peakDbfs)} dBFS"
        val transcript = update.snapshot.transcript
        if (transcript.isNotBlank()) lines += "识别文字：$transcript"
        else if (update.snapshot.asrUnavailable) lines += "识别文字：ASR 未就绪"
        if (update.testing) {
            lines += "测试状态：正在收音，请说话…"
        } else {
            val result = requireNotNull(update.result)
            lines += "测试结果：${result.status}"
            lines += result.guidance
        }
        audioInputProbeStatus.text = lines.joinToString("\n")
    }

    private fun playAudioInputRecording() {
        val recording = audioInputRecording ?: return
        audioInputProbeButton.isEnabled = false
        audioInputPlaybackButton.isEnabled = false
        audioInputPlaybackButton.text = "正在播放…"
        if (!recording.play { failure ->
                runOnUiThread {
                    audioInputRecording = null
                    audioInputPlaybackButton.visibility = View.GONE
                    audioInputPlaybackButton.text = "播放本次测试录音"
                    audioInputProbeButton.isEnabled = true
                    if (failure != null && !isFinishing && !isDestroyed) {
                        Toast.makeText(this, failure, Toast.LENGTH_LONG).show()
                    }
                }
            }) {
            clearAudioInputRecording()
            audioInputProbeButton.isEnabled = true
        }
    }

    private fun clearAudioInputRecording() {
        audioInputRecording?.clear()
        audioInputRecording = null
        if (::audioInputPlaybackButton.isInitialized) {
            audioInputPlaybackButton.visibility = View.GONE
            audioInputPlaybackButton.isEnabled = true
            audioInputPlaybackButton.text = "播放本次测试录音"
        }
    }

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
        private const val REQUEST_AUDIO_INPUT_PROBE = 211
        private val DIAGNOSTIC_TIME = java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
    }
}
