package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class AudioCaptureService : Service() {
    private lateinit var settings: SecureSettings
    private lateinit var recorder: AudioRecorder
    private lateinit var connectivity: ConnectivityMonitor
    private lateinit var audioManager: AudioManager
    private lateinit var worker: ExecutorService
    private lateinit var replyWorker: ExecutorService
    private lateinit var pendingReplies: PendingReplyStore
    private lateinit var tts: AndroidTtsController
    private val stopping = AtomicBoolean(false)
    private val replyPolling = AtomicBoolean(false)
    private var captureId = ""
    @Volatile private var activeEnrollmentSessionId = ""
    @Volatile private var pendingEnrollmentSessionId = ""
    @Volatile private var enrollmentOnly = false
    @Volatile private var modelPipeline: NativeModelPipeline? = null

    private val audioDeviceCallback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>) {
            recorder.restartInput()
        }

        override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>) {
            recorder.restartInput()
        }
    }

    override fun onCreate() {
        super.onCreate()
        settings = SecureSettings(this)
        tts = AndroidTtsController(this)
        worker = Executors.newSingleThreadExecutor { task -> Thread(task, "android-runtime-bridge") }
        replyWorker = Executors.newSingleThreadExecutor { task -> Thread(task, "device-reply-poll") }
        pendingReplies = PendingReplyStore(this)
        recorder = AudioRecorder(
            context = this,
            sink = PcmFrameSink { pcm16, sampleCount, capturedAtNanos ->
                NativeAudioState.addCapturedSamples(sampleCount)
                NativeAudioState.markAudioLevel(AudioLevelMeter.measure(pcm16, sampleCount))
                modelPipeline?.accept(pcm16, sampleCount, capturedAtNanos)
            },
            onFailure = ::handleRecordingFailure,
        )
        audioManager = getSystemService(AudioManager::class.java)
        audioManager.registerAudioDeviceCallback(audioDeviceCallback, null)
        createNotificationChannel()
        connectivity = ConnectivityMonitor(this, ::handleNetworkChanged).also { it.start() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action ?: ACTION_START) {
            ACTION_START -> startCaptureFromVisibleApp()
            ACTION_STOP -> stopCaptureAndService()
            ACTION_PAUSE_TTS -> pauseForTts()
            ACTION_RESUME_TTS -> resumeAfterTts()
            ACTION_START_ENROLLMENT -> startEnrollmentFromVisibleApp(
                intent?.getStringExtra(EXTRA_ENROLLMENT_SESSION_ID).orEmpty(),
            )
            ACTION_CANCEL_ENROLLMENT -> cancelEnrollment()
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        recorder.stop()
        modelPipeline?.close()
        modelPipeline = null
        connectivity.stop()
        tts.shutdown()
        runCatching { audioManager.unregisterAudioDeviceCallback(audioDeviceCallback) }
        worker.shutdown()
        replyWorker.shutdownNow()
        if (NativeAudioState.snapshot().running) NativeAudioState.markIdle()
        super.onDestroy()
    }

    private fun startCaptureFromVisibleApp() {
        if (NativeAudioState.snapshot().running) {
            if (activeEnrollmentSessionId.isNotBlank()) enrollmentOnly = false
            updateNotification()
            return
        }
        startAudioRuntime(startAmbientCapture = true)
    }

    private fun startEnrollmentFromVisibleApp(rawSessionId: String) {
        val sessionId = rawSessionId.trim().take(120)
        if (sessionId.isBlank()) {
            failAndStop("声纹录入 session 无效")
            return
        }
        val pipeline = modelPipeline
        if (NativeAudioState.snapshot().running && pipeline != null) {
            enrollmentOnly = false
            activeEnrollmentSessionId = sessionId
            pipeline.startEnrollment(sessionId)
            updateNotification()
            return
        }
        if (NativeAudioState.snapshot().state == "starting") {
            pendingEnrollmentSessionId = sessionId
            return
        }
        startAudioRuntime(startAmbientCapture = false, enrollmentSessionId = sessionId)
    }

    private fun startAudioRuntime(startAmbientCapture: Boolean, enrollmentSessionId: String = "") {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            failAndStop("麦克风权限未授予")
            return
        }
        if (!settings.isConfigured()) {
            failAndStop("请先配置 DeepSeek API key")
            return
        }
        NativeAudioState.markStarting()
        startMicrophoneForeground(notification("正在启动本地运行时"))
        worker.execute {
            runCatching {
                val staticDir = StaticAssets.extract(this).absolutePath
                PythonRuntime.start(settings.runtimeConfig(staticDir))
                PythonRuntime.setNetworkState(NativeAudioState.snapshot().networkOnline)
                val installer = ModelPackInstaller(this)
                val pack = checkNotNull(installer.current()) { "请先安装 Android 本地模型" }
                ModelPackState.markExisting(pack.version)
                val selfTest = ModelSelfTestState.restore(this, pack.version)
                check(selfTest.state == "ok") { "请先在设置页完成五项模型自检" }
                captureId = if (startAmbientCapture) PythonRuntime.startCapture(settings.ownerId()) else ""
                modelPipeline = NativeModelPipeline(
                    context = this,
                    pack = pack,
                    ownerId = settings.ownerId(),
                    captureId = captureId,
                    onWakeDetected = { Handler(Looper.getMainLooper()).post { tts.speak("我在") } },
                    onPartial = { text ->
                        if (text.isBlank()) NativeAudioState.clearPartial() else NativeAudioState.markPartial(text)
                    },
                    onEnrollmentProgress = ::handleEnrollmentProgress,
                    onReplyQueued = ::handleReplyQueued,
                    onFailure = ::handleModelFailure,
                )
                NativeAudioState.markRecording(captureId)
                recorder.start()
                val requestedEnrollment = enrollmentSessionId.ifBlank { pendingEnrollmentSessionId }
                pendingEnrollmentSessionId = ""
                if (requestedEnrollment.isNotBlank()) {
                    enrollmentOnly = !startAmbientCapture
                    activeEnrollmentSessionId = requestedEnrollment
                    modelPipeline?.startEnrollment(requestedEnrollment)
                }
                updateNotification()
            }.onFailure(::handleStartFailure)
        }
    }

    private fun stopCaptureAndService() {
        if (!stopping.compareAndSet(false, true)) return
        NativeAudioState.markStopping()
        recorder.stop()
        modelPipeline?.close()
        modelPipeline = null
        val enrollmentSessionId = activeEnrollmentSessionId
        activeEnrollmentSessionId = ""
        pendingEnrollmentSessionId = ""
        enrollmentOnly = false
        updateNotification()
        val stoppedCaptureId = captureId
        captureId = ""
        worker.execute {
            if (enrollmentSessionId.isNotBlank()) {
                runCatching { PythonRuntime.cancelSpeakerEnrollment(settings.ownerId(), enrollmentSessionId) }
            }
            if (stoppedCaptureId.isNotBlank()) {
                runCatching { PythonRuntime.stopCapture(settings.ownerId(), stoppedCaptureId) }
                    .onFailure { NativeAudioState.markError(it.message ?: "停止 capture 失败") }
            }
            NativeAudioState.markIdle()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun pauseForTts() {
        if (NativeAudioState.snapshot().state != "recording") return
        recorder.pause()
        NativeAudioState.markPausedForTts()
        updateNotification()
    }

    private fun resumeAfterTts() {
        if (NativeAudioState.snapshot().state != "paused_tts") return
        recorder.resume()
        NativeAudioState.markResumed()
        updateNotification()
    }

    private fun handleNetworkChanged(online: Boolean) {
        NativeAudioState.setNetworkOnline(online)
        if (!worker.isShutdown) {
            worker.execute {
                runCatching { PythonRuntime.setNetworkState(online) }
                if (online) startReplyPolling()
            }
        }
        if (NativeAudioState.snapshot().running) updateNotification()
    }

    private fun handleRecordingFailure(error: Throwable) {
        NativeAudioState.markError(error.message ?: "录音失败")
        updateNotification()
        val stoppedCaptureId = captureId
        captureId = ""
        worker.execute {
            if (stoppedCaptureId.isNotBlank()) {
                runCatching { PythonRuntime.stopCapture(settings.ownerId(), stoppedCaptureId) }
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun handleModelFailure(error: Throwable) {
        NativeAudioState.markError(error.message ?: "本地模型处理失败")
        recorder.stop()
        modelPipeline = null
        updateNotification()
        val stoppedCaptureId = captureId
        captureId = ""
        worker.execute {
            if (stoppedCaptureId.isNotBlank()) {
                runCatching { PythonRuntime.stopCapture(settings.ownerId(), stoppedCaptureId) }
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun handleReplyQueued(eventId: String) {
        pendingReplies.markPending(eventId)
        if (NativeAudioState.snapshot().networkOnline) startReplyPolling()
    }

    private fun handleEnrollmentProgress(
        state: String,
        sessionId: String,
        sampleCount: Int,
        sampleTotal: Int,
        error: String,
    ) {
        NativeAudioState.markEnrollment(state, sessionId, sampleCount, sampleTotal, error)
        updateNotification()
        if (state == "completed") {
            activeEnrollmentSessionId = ""
            if (enrollmentOnly) stopCaptureAndService()
        }
    }

    private fun cancelEnrollment() {
        val sessionId = activeEnrollmentSessionId.ifBlank {
            NativeAudioState.snapshot().enrollmentSessionId
        }
        modelPipeline?.cancelEnrollment()
        activeEnrollmentSessionId = ""
        NativeAudioState.clearEnrollment("cancelled")
        if (!worker.isShutdown && sessionId.isNotBlank()) {
            worker.execute {
                runCatching { PythonRuntime.cancelSpeakerEnrollment(settings.ownerId(), sessionId) }
                    .onFailure { NativeAudioState.markEnrollment("error", sessionId, 0, 3, it.message.orEmpty()) }
            }
        }
        updateNotification()
        if (enrollmentOnly) stopCaptureAndService()
    }

    private fun startReplyPolling() {
        if (replyWorker.isShutdown || !replyPolling.compareAndSet(false, true)) return
        replyWorker.execute {
            try {
                repeat(REPLY_POLL_ATTEMPTS) {
                    val pendingIds = pendingReplies.pendingIds()
                    if (pendingIds.isEmpty() || !NativeAudioState.snapshot().networkOnline) return@execute
                    val queue = runCatching { PythonRuntime.queueStatus(settings.ownerId()) }.getOrNull()
                    val events = queue?.optJSONArray("events")
                    if (events != null) {
                        for (index in 0 until events.length()) {
                            val event = events.optJSONObject(index) ?: continue
                            val eventId = event.optString("event_id")
                            if (eventId !in pendingIds) continue
                            when (event.optString("status")) {
                                "completed" -> {
                                    val reply = event.optJSONObject("dispatch")
                                        ?.optJSONObject("result")
                                        ?.optString("reply")
                                        .orEmpty()
                                    if (reply.isNotBlank()) {
                                        pendingReplies.markCompleted(eventId, reply)
                                        notifyReplyReady()
                                    } else {
                                        pendingReplies.removePending(eventId)
                                    }
                                }
                                "failed" -> pendingReplies.removePending(eventId)
                            }
                        }
                    }
                    if (pendingReplies.pendingIds().isEmpty()) return@execute
                    Thread.sleep(REPLY_POLL_INTERVAL_MILLIS)
                }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            } finally {
                replyPolling.set(false)
            }
        }
    }

    private fun notifyReplyReady() {
        val openIntent = PendingIntent.getActivity(
            this,
            2,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(getString(R.string.reply_ready_title))
            .setContentText(getString(R.string.reply_ready_text))
            .setContentIntent(openIntent)
            .setAutoCancel(true)
            .build()
        getSystemService(NotificationManager::class.java).notify(REPLY_NOTIFICATION_ID, notification)
    }

    private fun handleStartFailure(error: Throwable) {
        NativeAudioState.markError(error.message ?: "本地运行时启动失败")
        updateNotification()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun failAndStop(message: String) {
        NativeAudioState.markError(message)
        startMicrophoneForeground(notification(message))
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun startMicrophoneForeground(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun updateNotification() {
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            notification(statusText(NativeAudioState.snapshot())),
        )
    }

    private fun notification(content: String): Notification {
        val openIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, AudioCaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle(getString(R.string.capture_notification_title))
            .setContentText(content)
            .setContentIntent(openIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(Notification.Action.Builder(null, getString(R.string.stop_capture), stopIntent).build())
            .build()
    }

    private fun statusText(snapshot: NativeAudioSnapshot): String = when {
        snapshot.state == "starting" -> "正在启动本地运行时"
        snapshot.state == "recording" && snapshot.enrollmentState in setOf("recording", "processing", "error") ->
            "声纹录入 ${snapshot.enrollmentSampleCount}/${snapshot.enrollmentSampleTotal}"
        snapshot.state == "recording" -> "持续收音中，原始 PCM 不会保存"
        snapshot.state == "paused_tts" -> "播报中，已暂停收音"
        snapshot.state == "stopping" -> "正在停止并释放麦克风"
        snapshot.state == "error" -> snapshot.lastError.ifBlank { "录音发生错误" }
        else -> "收音待机未启动"
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.capture_notification_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.capture_notification_description)
            setSound(null, null)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "continuous_microphone"
        private const val NOTIFICATION_ID = 1001
        private const val REPLY_NOTIFICATION_ID = 1003
        private const val REPLY_POLL_ATTEMPTS = 120
        private const val REPLY_POLL_INTERVAL_MILLIS = 500L
        const val ACTION_START = "com.aiglasses.memoryassistant.action.START_CAPTURE"
        const val ACTION_STOP = "com.aiglasses.memoryassistant.action.STOP_CAPTURE"
        const val ACTION_PAUSE_TTS = "com.aiglasses.memoryassistant.action.PAUSE_FOR_TTS"
        const val ACTION_RESUME_TTS = "com.aiglasses.memoryassistant.action.RESUME_AFTER_TTS"
        const val ACTION_START_ENROLLMENT = "com.aiglasses.memoryassistant.action.START_ENROLLMENT"
        const val ACTION_CANCEL_ENROLLMENT = "com.aiglasses.memoryassistant.action.CANCEL_ENROLLMENT"
        private const val EXTRA_ENROLLMENT_SESSION_ID = "enrollment_session_id"

        fun start(context: Context) {
            val intent = Intent(context, AudioCaptureService::class.java).setAction(ACTION_START)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.startService(Intent(context, AudioCaptureService::class.java).setAction(ACTION_STOP))
        }

        fun startEnrollment(context: Context, sessionId: String) {
            val intent = Intent(context, AudioCaptureService::class.java)
                .setAction(ACTION_START_ENROLLMENT)
                .putExtra(EXTRA_ENROLLMENT_SESSION_ID, sessionId)
            context.startForegroundService(intent)
        }

        fun cancelEnrollment(context: Context) {
            if (NativeAudioState.snapshot().running) {
                context.startService(Intent(context, AudioCaptureService::class.java).setAction(ACTION_CANCEL_ENROLLMENT))
            }
        }

        fun pauseForTts(context: Context) {
            if (NativeAudioState.snapshot().running) {
                context.startService(Intent(context, AudioCaptureService::class.java).setAction(ACTION_PAUSE_TTS))
            }
        }

        fun resumeAfterTts(context: Context) {
            if (NativeAudioState.snapshot().running) {
                context.startService(Intent(context, AudioCaptureService::class.java).setAction(ACTION_RESUME_TTS))
            }
        }
    }
}
