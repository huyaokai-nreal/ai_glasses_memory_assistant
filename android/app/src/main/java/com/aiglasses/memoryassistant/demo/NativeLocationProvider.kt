package com.aiglasses.memoryassistant.demo

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.CancellationSignal
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import org.json.JSONObject
import java.io.Closeable
import java.util.concurrent.atomic.AtomicLong

internal data class NativeLocationResult(
    val status: String,
    val latitude: Double? = null,
    val longitude: Double? = null,
    val accuracy: Double? = null,
    val timestampSeconds: Double? = null,
    val error: String = "",
) {
    fun toJson(): JSONObject = JSONObject()
        .put("status", status)
        .put("source", SOURCE)
        .apply {
            latitude?.let { put("latitude", it) }
            longitude?.let { put("longitude", it) }
            accuracy?.let { put("accuracy", it) }
            timestampSeconds?.let { put("timestamp", it) }
            if (error.isNotBlank()) put("error", error)
        }

    companion object {
        const val SOURCE = "android_location_manager"

        fun available(location: Location): NativeLocationResult = NativeLocationResult(
            status = "available",
            latitude = location.latitude,
            longitude = location.longitude,
            accuracy = if (location.hasAccuracy()) location.accuracy.toDouble() else null,
            timestampSeconds = location.time.takeIf { it > 0 }?.div(1_000.0),
        )

        fun failure(status: String, error: String): NativeLocationResult = NativeLocationResult(
            status = status,
            error = error,
        )
    }
}

internal class NativeLocationProvider(
    context: Context,
    private val timeoutMillis: Long = DEFAULT_TIMEOUT_MILLIS,
    private val freshLocationMillis: Long = FRESH_LOCATION_MILLIS,
) : Closeable {
    private data class PendingRequest(
        val callback: (NativeLocationResult) -> Unit,
        val timeout: Runnable,
        val cancellationSignal: CancellationSignal? = null,
        val listener: LocationListener? = null,
    )

    private val appContext = context.applicationContext
    private val locationManager = appContext.getSystemService(LocationManager::class.java)
    private val handler = Handler(Looper.getMainLooper())
    private val requestIds = AtomicLong(0L)
    private val lock = Any()
    private val pending = mutableMapOf<Long, PendingRequest>()
    @Volatile private var closed = false

    fun request(callback: (NativeLocationResult) -> Unit) {
        if (closed) {
            callback(NativeLocationResult.failure("unavailable", "location_provider_closed"))
            return
        }
        if (!hasLocationPermission()) {
            callback(NativeLocationResult.failure("denied", "location_permission_denied"))
            return
        }
        if (!isLocationEnabled()) {
            callback(NativeLocationResult.failure("disabled", "system_location_disabled"))
            return
        }
        freshLastLocation()?.let {
            callback(NativeLocationResult.available(it))
            return
        }
        val provider = currentProvider()
        if (provider.isNullOrBlank()) {
            callback(NativeLocationResult.failure("unavailable", "location_provider_unavailable"))
            return
        }
        val requestId = requestIds.incrementAndGet()
        val timeout = Runnable {
            finish(requestId, NativeLocationResult.failure("timeout", "location_acquisition_timeout"))
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val signal = CancellationSignal()
                synchronized(lock) {
                    if (closed) {
                        callback(NativeLocationResult.failure("unavailable", "location_provider_closed"))
                        return
                    }
                    pending[requestId] = PendingRequest(callback, timeout, cancellationSignal = signal)
                }
                handler.postDelayed(timeout, timeoutMillis)
                locationManager.getCurrentLocation(provider, signal, appContext.mainExecutor) { location ->
                    finish(
                        requestId,
                        location?.let(NativeLocationResult::available)
                            ?: NativeLocationResult.failure("unavailable", "location_result_unavailable"),
                    )
                }
            } else {
                val listener = object : LocationListener {
                    override fun onLocationChanged(location: Location) {
                        finish(requestId, NativeLocationResult.available(location))
                    }

                    override fun onProviderDisabled(provider: String) {
                        finish(requestId, NativeLocationResult.failure("disabled", "system_location_disabled"))
                    }

                    override fun onProviderEnabled(provider: String) = Unit
                    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
                }
                synchronized(lock) {
                    if (closed) {
                        callback(NativeLocationResult.failure("unavailable", "location_provider_closed"))
                        return
                    }
                    pending[requestId] = PendingRequest(callback, timeout, listener = listener)
                }
                handler.postDelayed(timeout, timeoutMillis)
                @Suppress("DEPRECATION")
                locationManager.requestSingleUpdate(provider, listener, Looper.getMainLooper())
            }
        } catch (_: SecurityException) {
            finish(requestId, NativeLocationResult.failure("denied", "location_permission_denied"))
        } catch (_: IllegalArgumentException) {
            finish(requestId, NativeLocationResult.failure("unavailable", "location_provider_unavailable"))
        } catch (_: RuntimeException) {
            finish(requestId, NativeLocationResult.failure("unavailable", "location_request_failed"))
        }
    }

    override fun close() {
        val requests = synchronized(lock) {
            closed = true
            pending.values.toList().also { pending.clear() }
        }
        requests.forEach(::cancel)
    }

    private fun finish(requestId: Long, result: NativeLocationResult) {
        val request = synchronized(lock) { pending.remove(requestId) } ?: return
        cancel(request)
        request.callback(result)
    }

    private fun cancel(request: PendingRequest) {
        handler.removeCallbacks(request.timeout)
        request.cancellationSignal?.cancel()
        request.listener?.let { runCatching { locationManager.removeUpdates(it) } }
    }

    private fun hasLocationPermission(): Boolean =
        appContext.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            appContext.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun isLocationEnabled(): Boolean = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        locationManager.isLocationEnabled
    } else {
        @Suppress("DEPRECATION")
        locationManager.getProviders(true).any { it != LocationManager.PASSIVE_PROVIDER }
    }

    private fun currentProvider(): String? {
        val enabled = locationManager.getProviders(true)
        val hasFine = appContext.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        return listOf("fused", LocationManager.NETWORK_PROVIDER)
            .firstOrNull { it in enabled }
            ?: LocationManager.GPS_PROVIDER.takeIf { hasFine && it in enabled }
            ?: enabled.firstOrNull { it != LocationManager.PASSIVE_PROVIDER }
    }

    private fun freshLastLocation(): Location? {
        val fineGranted = appContext.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        val coarseGranted = appContext.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        if (!fineGranted && !coarseGranted) return null
        val nowElapsedNanos = SystemClock.elapsedRealtimeNanos()
        val nowWallMillis = System.currentTimeMillis()
        return locationManager.getProviders(true)
            .asSequence()
            .mapNotNull { provider -> runCatching { locationManager.getLastKnownLocation(provider) }.getOrNull() }
            .map { location ->
                val ageMillis = if (location.elapsedRealtimeNanos > 0L) {
                    ((nowElapsedNanos - location.elapsedRealtimeNanos).coerceAtLeast(0L) / 1_000_000L)
                } else {
                    (nowWallMillis - location.time).coerceAtLeast(0L)
                }
                location to ageMillis
            }
            .filter { (_, ageMillis) -> ageMillis <= freshLocationMillis }
            .minWithOrNull(
                compareBy<Pair<Location, Long>> { it.second }
                    .thenBy { if (it.first.hasAccuracy()) it.first.accuracy else Float.MAX_VALUE },
            )
            ?.first
    }

    companion object {
        const val DEFAULT_TIMEOUT_MILLIS = 8_000L
        const val FRESH_LOCATION_MILLIS = 15_000L
    }
}
