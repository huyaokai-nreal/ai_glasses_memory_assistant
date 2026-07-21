package com.aiglasses.memoryassistant.demo

import android.app.Activity
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
        val modelStatus = TextView(this).apply {
            text = if (installedVersion == null) "本地模型：未安装" else "本地模型：$installedVersion"
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
            addView(installModels, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(save, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        setContentView(ScrollView(this).apply { addView(content) })
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
}
