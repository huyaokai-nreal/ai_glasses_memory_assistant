package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest

class ConnectivityMonitor(
    context: Context,
    private val onChanged: (Boolean) -> Unit,
) {
    private val manager = context.getSystemService(ConnectivityManager::class.java)
    private var registered = false
    private val callback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) = publish()
        override fun onLost(network: Network) = publish()
        override fun onCapabilitiesChanged(network: Network, capabilities: NetworkCapabilities) = publish()
    }

    fun start() {
        if (registered) return
        registered = true
        manager.registerNetworkCallback(
            NetworkRequest.Builder().addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET).build(),
            callback,
        )
        publish()
    }

    fun stop() {
        if (!registered) return
        registered = false
        runCatching { manager.unregisterNetworkCallback(callback) }
    }

    private fun publish() {
        val network = manager.activeNetwork
        val capabilities = network?.let(manager::getNetworkCapabilities)
        val online = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
            && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        onChanged(online)
    }
}
