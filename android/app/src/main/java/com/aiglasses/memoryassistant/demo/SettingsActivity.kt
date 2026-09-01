package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.widget.TextView
import android.widget.Toast
import java.util.concurrent.CancellationException
import java.util.Locale

private data class LlmPreset(val provider: String, val model: String, val baseUrl: String, val apiKey: String)

class SettingsActivity : Activity() {
    private lateinit var settings: SecureSettings
    private var diagnosticExportInProgress = false
    private var audioInputProbe: AudioInputProbe? = null
    private var audioInputRecording: AudioInputProbeRecording? = null
    private var pendingAudioInputProbe = false
    private lateinit var audioInputProbeStatus: TextView
    private lateinit var audioInputProbeButton: Button
    private lateinit var audioInputPlaybackButton: Button
    private var offlineAudioUri: Uri? = null
    private var offlineAudioTest: OfflineAudioTestRunner? = null
    private var offlineAudioTranscript = ""
    private var offlineAudioEvalExport = ""
    private lateinit var offlineAudioStatus: TextView
    private lateinit var offlineAudioGainEnabled: CheckBox
    private lateinit var offlineAudioGainSlider: SeekBar
    private lateinit var offlineAudioGainLabel: TextView
    private lateinit var offlineAudioSelectButton: Button
    private lateinit var offlineAudioRunButton: Button
    private lateinit var offlineAudioCancelButton: Button
    private lateinit var offlineAudioCopyButton: Button
    private lateinit var offlineAudioCopyEvalButton: Button
    private lateinit var offlineAudioResult: TextView

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
        val llmPresets = listOf(
            LlmPreset("llama_cpp", "qwen3.8-27b-32k", "http://10.252.17.5:11438/v1", "ollama"),
            LlmPreset("deepseek", "deepseek-chat", "https://api.deepseek.com", ""),
        )
        val presetLabels = listOf("本地 Qwen（局域网）", "DeepSeek（云端）")
        val presetSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@SettingsActivity, android.R.layout.simple_spinner_item, presetLabels).also {
                it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>, view: android.view.View?, position: Int, id: Long) {
                    llmPresets[position].let { p ->
                        provider.setText(p.provider)
                        model.setText(p.model)
                        baseUrl.setText(p.baseUrl)
                        apiKey.setText(p.apiKey)
                    }
                }
                override fun onNothingSelected(parent: AdapterView<*>) {}
            }
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
                    )
                }.onSuccess {
                    Toast.makeText(this@SettingsActivity, "配置已保存", Toast.LENGTH_SHORT).show()
                    finish()
                }.onFailure { error ->
                    Toast.makeText(this@SettingsActivity, error.message ?: "配置无效", Toast.LENGTH_LONG).show()
                }
            }
        }
        val selfTestModels = Button(this).apply {
            text = "运行模型自检"
            styleActionButton()
            isEnabled = installedVersion != null
        }
        selfTestModels.setOnClickListener {
            if (NativeAudioState.snapshot().running) {
                Toast.makeText(this@SettingsActivity, "请先在首页停止全天收音，再运行模型自检", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            val liveVersion = ModelPackInstaller(this@SettingsActivity).currentVersion()
            if (liveVersion == null) {
                Toast.makeText(this@SettingsActivity, "未检测到模型包：请确认已运行 install_local_model_pack.sh，并完全关闭重开 App", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
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
                        version = liveVersion,
                        checkedAtMillis = System.currentTimeMillis(),
                        components = errorComponent(error),
                    ).also { ModelSelfTestState.save(this@SettingsActivity, it) }
                }
                runCatching { PythonRuntime.setDeviceState(org.json.JSONObject(NativeAudioState.snapshotJson())) }
                runOnUiThread {
                    modelStatus.text = modelStatusText(liveVersion, result)
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
            text = "导出诊断包"
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
        offlineAudioStatus = TextView(this).apply {
            text = "选择 PCM16 WAV 或 AAC M4A 后，可直接测试本机 VAD 与 SenseVoice；不会写入记忆或审计"
            setPadding(0, dp(12), 0, dp(8))
        }
        offlineAudioGainEnabled = CheckBox(this).apply {
            text = "使用增益"
            isChecked = false
            setOnCheckedChangeListener { _, _ ->
                updateOfflineAudioGainLabel()
                updateOfflineAudioControls()
            }
        }
        offlineAudioGainLabel = TextView(this).apply {
            setPadding(0, 0, 0, dp(4))
        }
        offlineAudioGainSlider = SeekBar(this).apply {
            max = OfflineAudioGain.MAX_DECIBELS - OfflineAudioGain.MIN_DECIBELS
            progress = OFFLINE_GAIN_ZERO_PROGRESS
            contentDescription = "离线音频增益"
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                    updateOfflineAudioGainLabel()
                }

                override fun onStartTrackingTouch(seekBar: SeekBar) = Unit

                override fun onStopTrackingTouch(seekBar: SeekBar) = Unit
            })
        }
        val offlineAudioGainResetButton = Button(this).apply {
            text = "恢复 0 dB"
            styleActionButton()
            setOnClickListener { offlineAudioGainSlider.progress = OFFLINE_GAIN_ZERO_PROGRESS }
        }
        offlineAudioSelectButton = Button(this).apply {
            text = "选择 WAV 或 M4A 文件"
            styleActionButton()
            setOnClickListener { selectOfflineAudioFile() }
        }
        offlineAudioRunButton = Button(this).apply {
            text = "运行离线 VAD/ASR 测试"
            styleActionButton(primary = true)
            setOnClickListener { startOfflineAudioTest() }
        }
        offlineAudioCancelButton = Button(this).apply {
            text = "取消离线测试"
            styleActionButton()
            visibility = View.GONE
            setOnClickListener { cancelOfflineAudioTest() }
        }
        offlineAudioCopyButton = Button(this).apply {
            text = "复制转写文本"
            styleActionButton()
            visibility = View.GONE
            setOnClickListener { copyOfflineAudioTranscript() }
        }
        offlineAudioCopyEvalButton = Button(this).apply {
            text = "复制 Eval_Ali 调试事件 JSON"
            styleActionButton()
            visibility = View.GONE
            setOnClickListener { copyOfflineAudioEvalExport() }
        }
        offlineAudioResult = TextView(this).apply {
            setTextIsSelectable(true)
            setPadding(dp(12), dp(12), dp(12), dp(12))
            setTextColor(Color.rgb(32, 33, 35))
            setBackgroundColor(Color.WHITE)
            visibility = View.GONE
        }
        updateOfflineAudioGainLabel()

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(24))
            setBackgroundColor(Color.rgb(246, 247, 249))
            addView(sectionTitle("账户与服务"))
            addView(TextView(this@SettingsActivity).apply {
                text = "快速预设（选完再点保存并返回）"
                textSize = 13f
                setTextColor(Color.rgb(75, 85, 99))
                setPadding(0, dp(8), 0, dp(4))
            })
            addView(presetSpinner, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(6)
            })
            addView(owner)
            addLabeledField("服务提供商", provider)
            addLabeledField("模型名称", model)
            addLabeledField("API 地址", baseUrl)
            addLabeledField("API 密钥", apiKey)
            addView(sectionTitle("本地语音模型"))
            addView(TextView(this@SettingsActivity).apply {
                text = "模型包用电脑端脚本装一次即可：android/tools/install_local_model_pack.sh --serial <设备序列>。装在 App 私有存储，之后重装 APK 会保留，无需重复安装。装完点下方「运行模型自检」。"
                textSize = 13f
                setTextColor(Color.rgb(75, 85, 99))
                setPadding(0, dp(8), 0, dp(4))
            })
            addView(modelStatus)
            addActionButton(selfTestModels)
            addView(sectionTitle("收音设备"))
            addView(audioInputProbeStatus)
            addActionButton(audioInputProbeButton)
            addActionButton(audioInputPlaybackButton)
            addView(sectionTitle("离线音频模型测试"))
            addView(offlineAudioStatus)
            addView(offlineAudioGainEnabled)
            addView(offlineAudioGainLabel)
            addView(offlineAudioGainSlider)
            addActionButton(offlineAudioGainResetButton)
            addActionButton(offlineAudioSelectButton)
            addActionButton(offlineAudioRunButton)
            addActionButton(offlineAudioCancelButton)
            addActionButton(offlineAudioCopyButton)
            addActionButton(offlineAudioCopyEvalButton)
            addView(offlineAudioResult, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(8)
            })
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
        updateOfflineAudioControls()
    }

    override fun onDestroy() {
        audioInputProbe?.cancel()
        audioInputProbe = null
        clearAudioInputRecording()
        discardOfflineAudioTest()
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
        discardOfflineAudioTest()
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
        if (requestCode == REQUEST_OFFLINE_AUDIO) {
            val uri = data?.data
            if (resultCode == RESULT_OK && uri != null) {
                offlineAudioUri = uri
                clearOfflineAudioResult()
                offlineAudioStatus.text = "已选择音频：${uri.lastPathSegment ?: "未命名文件"}"
                updateOfflineAudioControls()
            }
            return
        }
        if (requestCode != REQUEST_DIAGNOSTIC_EXPORT) return
        val destination = data?.data
        if (resultCode != RESULT_OK || destination == null) return
        if (diagnosticExportInProgress) return
        diagnosticExportInProgress = true
        Thread({
            val result = runCatching { DiagnosticExporter(this).exportPlain(destination) }
            runOnUiThread {
                diagnosticExportInProgress = false
                result.onSuccess {
                    Toast.makeText(this, "诊断包已导出（未加密 .zip）", Toast.LENGTH_LONG).show()
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

    private fun selectOfflineAudioFile() {
        if (NativeAudioState.snapshot().running) {
            Toast.makeText(this, "请先停止全天收音", Toast.LENGTH_LONG).show()
            return
        }
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                type = "audio/*"
            },
            REQUEST_OFFLINE_AUDIO,
        )
    }

    private fun startOfflineAudioTest() {
        if (NativeAudioState.snapshot().running) {
            Toast.makeText(this, "请先停止全天收音", Toast.LENGTH_LONG).show()
            return
        }
        val uri = offlineAudioUri ?: run {
            Toast.makeText(this, "请先选择 WAV 或 M4A 文件", Toast.LENGTH_LONG).show()
            return
        }
        if (offlineAudioTest != null) return
        val gainEnabled = offlineAudioGainEnabled.isChecked
        val gainDecibels = offlineAudioGainDecibels()
        clearOfflineAudioResult()
        val runner = OfflineAudioTestRunner(applicationContext)
        offlineAudioTest = runner
        offlineAudioStatus.text = "正在运行本机 VAD 与 SenseVoice，请稍候…"
        updateOfflineAudioControls()
        Thread({
            val result = runCatching {
                runner.run(
                    uri = uri,
                    gainEnabled = gainEnabled,
                    gainDecibels = gainDecibels,
                )
            }
            runOnUiThread {
                if (offlineAudioTest !== runner || isFinishing || isDestroyed) return@runOnUiThread
                offlineAudioTest = null
                result.onSuccess { completed ->
                    renderOfflineAudioResult(completed)
                    offlineAudioStatus.text = "离线音频测试完成；结果仅保留在当前设置页"
                }.onFailure { error ->
                    offlineAudioStatus.text = when (error) {
                        is CancellationException -> "离线音频测试已取消"
                        else -> "离线音频测试失败：${error.message ?: "未知错误"}"
                    }
                }
                updateOfflineAudioControls()
            }
        }, "offline-wav-asr-test").start()
    }

    private fun cancelOfflineAudioTest() {
        offlineAudioTest?.cancel() ?: return
        offlineAudioStatus.text = "正在取消离线音频测试…"
        offlineAudioCancelButton.isEnabled = false
    }

    private fun discardOfflineAudioTest() {
        offlineAudioTest?.cancel()
        offlineAudioTest = null
        offlineAudioUri = null
        offlineAudioTranscript = ""
        offlineAudioEvalExport = ""
        if (::offlineAudioResult.isInitialized) {
            offlineAudioResult.text = ""
            offlineAudioResult.visibility = View.GONE
            offlineAudioCopyButton.visibility = View.GONE
            offlineAudioCopyEvalButton.visibility = View.GONE
        }
    }

    private fun clearOfflineAudioResult() {
        offlineAudioTranscript = ""
        offlineAudioEvalExport = ""
        offlineAudioResult.text = ""
        offlineAudioResult.visibility = View.GONE
        offlineAudioCopyButton.visibility = View.GONE
        offlineAudioCopyEvalButton.visibility = View.GONE
    }

    private fun updateOfflineAudioGainLabel() {
        if (!::offlineAudioGainLabel.isInitialized) return
        val decibels = offlineAudioGainDecibels()
        offlineAudioGainLabel.text = if (offlineAudioGainEnabled.isChecked) {
            "当前增益：${if (decibels >= 0) "+" else ""}$decibels dB"
        } else {
            "当前使用原音频；滑杆设为 ${if (decibels >= 0) "+" else ""}$decibels dB 但不会生效"
        }
    }

    private fun updateOfflineAudioControls() {
        if (!::offlineAudioRunButton.isInitialized) return
        val audioRunning = NativeAudioState.snapshot().running
        val testing = offlineAudioTest != null
        val modelAvailable = ModelPackInstaller(this).currentVersion() != null
        offlineAudioSelectButton.isEnabled = !audioRunning && !testing
        offlineAudioGainEnabled.isEnabled = !audioRunning && !testing
        offlineAudioGainSlider.isEnabled = !audioRunning && !testing && offlineAudioGainEnabled.isChecked
        offlineAudioRunButton.isEnabled = !audioRunning && !testing && modelAvailable && offlineAudioUri != null
        offlineAudioCancelButton.visibility = if (testing) View.VISIBLE else View.GONE
        offlineAudioCancelButton.isEnabled = testing
    }

    private fun offlineAudioGainDecibels(): Int =
        offlineAudioGainSlider.progress + OfflineAudioGain.MIN_DECIBELS

    private fun renderOfflineAudioResult(result: OfflineAudioTestResult) {
        offlineAudioTranscript = result.transcript
        val caseId = offlineAudioUri?.lastPathSegment
            ?.substringBeforeLast('.', missingDelimiterValue = "")
            ?.takeIf { it.matches(Regex("[A-Za-z0-9._-]{1,128}")) }
        offlineAudioEvalExport = caseId?.let(result::toEvalAliDebugJson).orEmpty()
        val gain = result.gain
        val lines = mutableListOf(
            "模型包：${result.modelVersion}",
            "输入：${result.sourceFormat} / ${result.sourceSampleRate} Hz / ${result.sourceChannelCount} 声道 / ${result.sourceDurationMillis / 1_000.0} 秒",
            "规范化：${OfflineWavPcm16Reader.TARGET_SAMPLE_RATE} Hz 单声道 / ${result.normalizedSampleCount} 样本",
            "增益：${if (gain.enabled) "${if (gain.decibels >= 0) "+" else ""}${gain.decibels} dB" else "未启用，使用原音频"}",
            "峰值：${String.format(Locale.US, "%.3f", gain.inputPeak)} -> ${String.format(Locale.US, "%.3f", gain.outputPeak)}",
            "处理耗时：${result.elapsedMillis} ms",
        )
        if (gain.clippedSampleCount > 0) lines += "警告：增益后有 ${gain.clippedSampleCount} 个样本削波"
        if (result.segments.isEmpty()) {
            lines += "VAD 未检测到语音片段"
        } else {
            lines += "VAD/ASR 片段："
            result.segments.forEachIndexed { index, segment ->
                val language = segment.language.takeIf(String::isNotBlank)?.let { " / $it" }.orEmpty()
                lines += "${index + 1}. ${segment.startMillis}–${segment.endMillis} ms$language：${segment.text.ifBlank { "（无文字）" }}"
            }
            lines += "合并转写：${result.transcript.ifBlank { "（无文字）" }}"
        }
        if (caseId == null) lines += "Eval_Ali 导出要求 WAV 文件名为合法 case_id；当前文件仅可查看转写。"
        offlineAudioResult.text = lines.joinToString("\n")
        offlineAudioResult.visibility = View.VISIBLE
        offlineAudioCopyButton.visibility = if (offlineAudioTranscript.isBlank()) View.GONE else View.VISIBLE
        offlineAudioCopyEvalButton.visibility = if (offlineAudioEvalExport.isBlank()) View.GONE else View.VISIBLE
    }

    private fun copyOfflineAudioTranscript() {
        if (offlineAudioTranscript.isBlank()) return
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("离线音频转写", offlineAudioTranscript))
        Toast.makeText(this, "已复制转写文本", Toast.LENGTH_SHORT).show()
    }

    private fun copyOfflineAudioEvalExport() {
        if (offlineAudioEvalExport.isBlank()) return
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("Eval_Ali Android 调试事件", offlineAudioEvalExport))
        Toast.makeText(this, "已复制 Eval_Ali 调试事件；不含原始音频", Toast.LENGTH_SHORT).show()
    }

    private fun requestDiagnosticExport() {
        if (NativeAudioState.snapshot().running) {
            Toast.makeText(this, "请先停止全天收音", Toast.LENGTH_LONG).show()
            return
        }
        val fileName = "ai-glasses-diagnostic-${java.time.LocalDateTime.now().format(DIAGNOSTIC_TIME)}.zip"
        startActivityForResult(
            Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "application/zip"
                putExtra(Intent.EXTRA_TITLE, fileName)
            },
            REQUEST_DIAGNOSTIC_EXPORT,
        )
    }

    companion object {
        private const val REQUEST_DIAGNOSTIC_EXPORT = 210
        private const val REQUEST_AUDIO_INPUT_PROBE = 211
        private const val REQUEST_OFFLINE_AUDIO = 212
        private const val OFFLINE_GAIN_ZERO_PROGRESS = -OfflineAudioGain.MIN_DECIBELS
        private val DIAGNOSTIC_TIME = java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
    }
}
