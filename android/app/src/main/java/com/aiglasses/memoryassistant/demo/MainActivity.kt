package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.webkit.CookieManager
import android.webkit.GeolocationPermissions
import android.webkit.WebResourceResponse
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebChromeClient
import android.webkit.WebViewClient
import android.widget.Toast
import org.json.JSONObject
import java.io.ByteArrayInputStream

class MainActivity : Activity() {
    private lateinit var settings: SecureSettings
    private lateinit var webView: WebView
    private lateinit var tts: AndroidTtsController
    private var loadedRuntimeUrl: String = ""
    private var settingsOpen = false
    private var pendingMicrophoneStart = false
    private var pendingEnrollmentSessionId = ""
    private var pendingGeolocationOrigin = ""
    private var pendingGeolocationCallback: GeolocationPermissions.Callback? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = SecureSettings(this)
        tts = AndroidTtsController(this)
        webView = WebView(this)
        configureWebView()
        setContentView(webView)
    }

    override fun onResume() {
        super.onResume()
        if (!settings.isConfigured()) {
            if (!settingsOpen) openSettings()
            return
        }
        settingsOpen = false
        if (loadedRuntimeUrl.isEmpty()) startRuntime()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menu.add(Menu.NONE, MENU_SETTINGS, Menu.NONE, getString(R.string.settings))
            .setShowAsAction(MenuItem.SHOW_AS_ACTION_NEVER)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == MENU_SETTINGS) {
            openSettings()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onDestroy() {
        finishGeolocationPermission(false)
        tts.shutdown()
        webView.destroy()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_LOCATION) {
            val granted = checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
                checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
            finishGeolocationPermission(granted)
            return
        }
        if (requestCode != REQUEST_MICROPHONE) return
        val granted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (granted && pendingEnrollmentSessionId.isNotBlank()) {
            AudioCaptureService.startEnrollment(this, pendingEnrollmentSessionId)
        } else if (granted && pendingMicrophoneStart) {
            AudioCaptureService.start(this)
        } else if (!granted) {
            NativeAudioState.markIdle()
            Toast.makeText(this, "需要麦克风权限才能收音或录入声纹", Toast.LENGTH_LONG).show()
        }
        pendingMicrophoneStart = false
        pendingEnrollmentSessionId = ""
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            setGeolocationEnabled(true)
        }
        webView.addJavascriptInterface(NativeAppBridge(this), JS_BRIDGE_NAME)
        webView.webChromeClient = object : WebChromeClient() {
            override fun onGeolocationPermissionsShowPrompt(
                origin: String?,
                callback: GeolocationPermissions.Callback?,
            ) {
                if (callback == null || !LoopbackOriginPolicy.allows(origin, loadedRuntimeUrl)) {
                    callback?.invoke(origin, false, false)
                    return
                }
                if (hasLocationPermission()) {
                    callback.invoke(origin, true, false)
                    return
                }
                finishGeolocationPermission(false)
                pendingGeolocationOrigin = origin.orEmpty()
                pendingGeolocationCallback = callback
                requestPermissions(
                    arrayOf(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION,
                    ),
                    REQUEST_LOCATION,
                )
            }
        }
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                if (isTrustedRuntimeUrl(request.url)) return false
                Toast.makeText(this@MainActivity, "已阻止离开本机页面", Toast.LENGTH_SHORT).show()
                return true
            }

            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                if (isTrustedRuntimeUrl(request.url)) return null
                return WebResourceResponse(
                    "text/plain",
                    Charsets.UTF_8.name(),
                    403,
                    "Blocked",
                    emptyMap(),
                    ByteArrayInputStream(ByteArray(0)),
                )
            }
        }
    }

    private fun startRuntime() {
        val staticDir = StaticAssets.extract(this).absolutePath
        runCatching { PythonRuntime.start(settings.runtimeConfig(staticDir)) }
            .onSuccess { endpoint ->
                val installedVersion = ModelPackInstaller(this).currentVersion()
                ModelPackState.markExisting(installedVersion)
                ModelSelfTestState.restore(this, installedVersion)
                runCatching { PythonRuntime.setDeviceState(JSONObject(NativeAudioState.snapshotJson())) }
                val cookies = CookieManager.getInstance()
                cookies.setAcceptCookie(true)
                cookies.setAcceptThirdPartyCookies(webView, false)
                val encodedToken = Uri.encode(endpoint.localToken)
                cookies.setCookie(
                    endpoint.baseUrl,
                    "ai_glasses_local_token=$encodedToken; Path=/; HttpOnly; SameSite=Strict",
                )
                cookies.flush()
                loadedRuntimeUrl = endpoint.baseUrl
                webView.loadUrl("${endpoint.baseUrl}/")
            }
            .onFailure { error ->
                Toast.makeText(this, error.message ?: "本地服务启动失败", Toast.LENGTH_LONG).show()
                openSettings()
            }
    }

    private fun openSettings() {
        settingsOpen = true
        startActivity(Intent(this, SettingsActivity::class.java))
    }

    fun requestMicrophoneAndStart() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            AudioCaptureService.start(this)
            return
        }
        pendingMicrophoneStart = true
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions += Manifest.permission.POST_NOTIFICATIONS
        }
        requestPermissions(permissions.toTypedArray(), REQUEST_MICROPHONE)
    }

    fun requestMicrophoneAndStartEnrollment(sessionId: String) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            AudioCaptureService.startEnrollment(this, sessionId)
            return
        }
        pendingMicrophoneStart = false
        pendingEnrollmentSessionId = sessionId
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions += Manifest.permission.POST_NOTIFICATIONS
        }
        requestPermissions(permissions.toTypedArray(), REQUEST_MICROPHONE)
    }

    fun speakWithSystemTts(text: String) = tts.speak(text)

    fun stopSystemTts() = tts.stop()

    fun openNativeSettings() = openSettings()

    private fun hasLocationPermission(): Boolean =
        checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun finishGeolocationPermission(granted: Boolean) {
        val callback = pendingGeolocationCallback ?: return
        val origin = pendingGeolocationOrigin
        pendingGeolocationCallback = null
        pendingGeolocationOrigin = ""
        callback.invoke(origin, granted, false)
    }

    private fun isTrustedRuntimeUrl(uri: Uri): Boolean {
        if (loadedRuntimeUrl.isBlank()) return false
        val expected = Uri.parse(loadedRuntimeUrl)
        return uri.scheme == expected.scheme
            && uri.host == expected.host
            && uri.port == expected.port
    }

    companion object {
        private const val MENU_SETTINGS = 1
        private const val REQUEST_MICROPHONE = 100
        private const val REQUEST_LOCATION = 101
        private const val JS_BRIDGE_NAME = "AiGlassesAndroid"
    }
}
