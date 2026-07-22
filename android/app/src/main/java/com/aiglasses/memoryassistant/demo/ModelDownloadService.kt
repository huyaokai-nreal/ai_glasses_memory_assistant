package com.aiglasses.memoryassistant.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class ModelDownloadService : Service() {
    private val active = AtomicBoolean(false)
    private val worker = Executors.newSingleThreadExecutor { task -> Thread(task, "model-pack-download") }

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val url = intent?.getStringExtra(EXTRA_MANIFEST_URL).orEmpty()
        if (!active.compareAndSet(false, true)) return START_NOT_STICKY
        startDataSyncForeground(notification("正在读取模型清单", 0, 0))
        worker.execute { download(url) }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        worker.shutdown()
        super.onDestroy()
    }

    private fun download(url: String) {
        val installer = ModelPackInstaller(this)
        runCatching {
            check(!NativeAudioState.snapshot().running) { "录音中不能更新模型，请先停止全天待机" }
            ModelPackState.markDownloading()
            val manifest = installer.fetchManifest(url)
            installer.install(manifest) { progress ->
                ModelPackState.markProgress(progress)
                val percent = if (progress.totalBytes > 0) {
                    (progress.downloadedBytes * 100 / progress.totalBytes).toInt().coerceIn(0, 100)
                } else 0
                getSystemService(NotificationManager::class.java).notify(
                    NOTIFICATION_ID,
                    notification("正在下载模型 ${progress.fileIndex}/${progress.fileCount}", percent, 100),
                )
            }
        }.onSuccess { pack ->
            ModelSelfTestState.clear(this, pack.version)
            ModelPackState.markInstalled(pack.version)
            getSystemService(NotificationManager::class.java).notify(
                NOTIFICATION_ID,
                notification("模型 ${pack.version} 已安装，请运行模型自检", 0, 0, ongoing = false),
            )
        }.onFailure { error ->
            ModelPackState.markError(error.message ?: "模型安装失败")
            getSystemService(NotificationManager::class.java).notify(
                NOTIFICATION_ID,
                notification(error.message ?: "模型安装失败", 0, 0, ongoing = false),
            )
        }
        stopForeground(STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    private fun startDataSyncForeground(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun notification(content: String, progress: Int, max: Int, ongoing: Boolean = true): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, SettingsActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(getString(R.string.model_notification_title))
            .setContentText(content)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            .setProgress(max, progress, max == 0 && ongoing)
            .build()
    }

    private fun createChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.model_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    companion object {
        private const val CHANNEL_ID = "model_download"
        private const val NOTIFICATION_ID = 1002
        private const val EXTRA_MANIFEST_URL = "manifest_url"

        fun start(context: Context, manifestUrl: String) {
            require(manifestUrl.startsWith("https://")) { "模型清单 URL 必须使用 HTTPS" }
            context.startForegroundService(
                Intent(context, ModelDownloadService::class.java)
                    .putExtra(EXTRA_MANIFEST_URL, manifestUrl),
            )
        }
    }
}
