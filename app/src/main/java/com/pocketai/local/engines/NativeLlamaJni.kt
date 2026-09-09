package com.pocketai.local.engines

import com.pocketai.core.logging.AppLogger

fun interface NativeTokenCallback {
    fun onToken(token: String)
}

/**
 * Direct JNI bridge to the PrismML llama.cpp native shared library.
 * Loads the compiled arm64-v8a / x86_64 binaries (libggml-base, libggml-cpu,
 * libggml, libllama, libpocketai_llama).
 */
object NativeLlamaJni {
    private const val TAG = "NativeLlamaJni"

    val isAvailable: Boolean

    init {
        var success = false
        try {
            System.loadLibrary("ggml-base")
            System.loadLibrary("ggml-cpu")
            System.loadLibrary("ggml")
            System.loadLibrary("llama")
            System.loadLibrary("pocketai_llama")
            nativeInitBackend()
            success = true
            AppLogger.i(TAG, "PrismML llama.cpp native libraries loaded successfully: ${nativeGetBackendInfo()}")
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Failed to load PrismML native libraries: ${t.message}", t)
            success = false
        }
        isAvailable = success
    }

    @JvmStatic
    external fun nativeInitBackend(): Boolean

    @JvmStatic
    external fun nativeGetBackendInfo(): String

    @JvmStatic
    external fun nativeLoadModel(
        modelPath: String,
        nGpuLayers: Int,
        useMmap: Boolean,
        errorHolder: Array<String?>
    ): Long

    @JvmStatic
    external fun nativeCreateContext(
        modelHandle: Long,
        nCtx: Int,
        nBatch: Int,
        nThreads: Int,
        errorHolder: Array<String?>
    ): Long

    @JvmStatic
    external fun nativeFreeContext(contextHandle: Long)

    @JvmStatic
    external fun nativeFreeModel(modelHandle: Long)

    @JvmStatic
    external fun nativeGetModelInfo(modelHandle: Long): String?

    @JvmStatic
    external fun nativeStopGeneration()

    @JvmStatic
    external fun nativeGenerateStream(
        contextHandle: Long,
        modelHandle: Long,
        prompt: String,
        temperature: Float,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
        maxTokens: Int,
        stopTokens: Array<String>?,
        callback: NativeTokenCallback
    ): Boolean
}
