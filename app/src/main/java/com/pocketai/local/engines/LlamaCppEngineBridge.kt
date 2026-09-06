package com.pocketai.local.engines

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import com.pocketai.core.error.PocketAIException
import com.pocketai.core.logging.AppLogger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.nehuatl.llamacpp.LlamaAndroid
import java.util.concurrent.atomic.AtomicInteger

/**
 * Production bridge for on-device GGUF inference powered by llama.cpp.
 * Bridges Android ContentResolver ParcelFileDescriptors directly to native
 * llama.cpp contexts without copying large model files into internal app storage.
 */
class LlamaCppEngineBridge(private val context: Context) : InferenceEngine {

    override val engineName: String = "llama.cpp GGUF Engine"

    private val engineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override val isNativeEngineAvailable: Boolean by lazy {
        try {
            Class.forName("org.nehuatl.llamacpp.LlamaAndroid")
            true
        } catch (e: Throwable) {
            AppLogger.w("LlamaCppEngineBridge", "Native inference engine unavailable: ${e.message}")
            false
        }
    }

    private var llamaAndroidInstance: LlamaAndroid? = null

    private fun getLlamaAndroid(): LlamaAndroid {
        return llamaAndroidInstance ?: synchronized(this) {
            llamaAndroidInstance ?: LlamaAndroid(context.contentResolver).also {
                llamaAndroidInstance = it
            }
        }
    }

    @Volatile
    private var currentContextId: Int? = null

    @Volatile
    private var activeModelName: String? = null

    @Volatile
    private var isCurrentlyLoaded: Boolean = false

    @Volatile
    private var currentTokenListener: ((String) -> Unit)? = null

    @Volatile
    private var lastLoadTimeMs: Long = 0L

    @Volatile
    private var latestGenerationDiagnostics: GenerationDiagnostics? = null

    @Volatile
    private var loadedModelDetails: Map<String, Any>? = null

    override fun isLoaded(): Boolean = isCurrentlyLoaded && currentContextId != null

    override fun getLatestGenerationDiagnostics(): GenerationDiagnostics? = latestGenerationDiagnostics

    override fun getHardwareDiagnostics(): HardwareDiagnostics {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)

        val rawTotal = if (memInfo.totalMem > 0) memInfo.totalMem else Runtime.getRuntime().maxMemory()
        val rawAvail = if (memInfo.availMem > 0) memInfo.availMem else Runtime.getRuntime().freeMemory()

        val totalMb = (rawTotal / (1024 * 1024)).coerceAtLeast(1024)
        val availMb = (rawAvail / (1024 * 1024)).coerceAtLeast(512)
        val cores = Runtime.getRuntime().availableProcessors()

        // Recommended max model size leaves at least 1.5GB RAM for Android OS & system apps
        val maxSafeModelMb = (availMb - 1500).coerceAtLeast(400)

        return HardwareDiagnostics(
            totalRamMb = totalMb,
            availableRamMb = availMb,
            isHardwareSufficient = availMb >= 1600,
            recommendedMaxModelSizeMb = maxSafeModelMb,
            processorCores = cores
        )
    }

    override suspend fun loadModelFromUri(
        uri: Uri,
        modelName: String,
        params: ModelInferenceParams
    ): Result<EngineStatus> = withContext(Dispatchers.IO) {
        try {
            if (!isNativeEngineAvailable) {
                return@withContext Result.failure(
                    PocketAIException.NativeEngineUnavailableException(
                        "llama.cpp native binary is not available on this device runtime."
                    )
                )
            }

            val diagnostics = getHardwareDiagnostics()
            AppLogger.i(
                "LlamaCppEngineBridge",
                "Initiating load for '$modelName' (URI: $uri). Free RAM: ${diagnostics.availableRamMb}MB"
            )

            val effectiveUri = when {
                uri.scheme == null || uri.scheme?.isEmpty() == true -> Uri.fromFile(java.io.File(uri.path ?: uri.toString()))
                else -> uri
            }

            // Open ParcelFileDescriptor in read mode via ContentResolver with fallback to File PFD
            val pfd = try {
                context.contentResolver.openFileDescriptor(effectiveUri, "r")
                    ?: if (effectiveUri.scheme == "file" && effectiveUri.path != null) {
                        android.os.ParcelFileDescriptor.open(
                            java.io.File(effectiveUri.path!!),
                            android.os.ParcelFileDescriptor.MODE_READ_ONLY
                        )
                    } else null
            } catch (e: Exception) {
                if (effectiveUri.scheme == "file" && effectiveUri.path != null) {
                    try {
                        android.os.ParcelFileDescriptor.open(
                            java.io.File(effectiveUri.path!!),
                            android.os.ParcelFileDescriptor.MODE_READ_ONLY
                        )
                    } catch (e2: Exception) {
                        AppLogger.e("LlamaCppEngineBridge", "Failed to open descriptor for $effectiveUri: ${e2.message}")
                        return@withContext Result.failure(
                            PocketAIException.StoragePermissionException(effectiveUri.toString())
                        )
                    }
                } else {
                    AppLogger.e("LlamaCppEngineBridge", "Failed to open descriptor for $effectiveUri: ${e.message}")
                    return@withContext Result.failure(
                        PocketAIException.StoragePermissionException(effectiveUri.toString())
                    )
                }
            } ?: return@withContext Result.failure(
                PocketAIException.ModelNotFoundException("Cannot open descriptor for $effectiveUri")
            )

            val fileSizeBytes = pfd.statSize
            val fileSizeMb = fileSizeBytes / (1024 * 1024)

            // Memory headroom validation before taking over memory
            if (fileSizeMb > diagnostics.availableRamMb) {
                pfd.close()
                return@withContext Result.failure(
                    PocketAIException.InsufficientMemoryException(
                        requiredMb = fileSizeMb,
                        availableMb = diagnostics.availableRamMb
                    )
                )
            }

            // If a previous model is loaded in memory, release it first
            if (isCurrentlyLoaded) {
                unload()
            }

            val loadStartTime = System.currentTimeMillis()

            // Detach the native file descriptor so the native llama.cpp engine can map it directly
            val fd = pfd.detachFd()

            val effectiveContextLength = if (params.contextLength > 0) params.contextLength else 2048
            val effectiveThreads = if (params.threads > 0) params.threads else diagnostics.processorCores.coerceIn(2, 6)

            val startParams = mutableMapOf<String, Any>(
                "model" to effectiveUri.toString(),
                "model_fd" to fd,
                "n_ctx" to effectiveContextLength,
                "n_batch" to 512,
                "n_threads" to effectiveThreads,
                "n_gpu_layers" to 0,
                "use_mmap" to true,
                "use_mlock" to false,
                "embedding" to false,
                "vocab_only" to false,
                "lora" to "",
                "lora_scaled" to 1.0,
                "rope_freq_base" to 0.0,
                "rope_freq_scale" to 0.0
            )

            val engine = getLlamaAndroid()
            try {
                engine.setContextLimit(4)
            } catch (e: Throwable) {
                // Ignore if not supported
            }

            val tokenForwarder: (String) -> Unit = { token ->
                currentTokenListener?.invoke(token)
            }

            val startResult = engine.startEngine(startParams, tokenForwarder)
            if (startResult == null || !startResult.containsKey("contextId")) {
                return@withContext Result.failure(
                    PocketAIException.ModelLoadFailedException(
                        modelName,
                        "Native llama.cpp engine failed to initialize context from GGUF weights."
                    )
                )
            }

            val contextId = (startResult["contextId"] as? Number)?.toInt()
                ?: return@withContext Result.failure(
                    PocketAIException.ModelLoadFailedException(
                        modelName,
                        "Engine returned an invalid native context identifier."
                    )
                )

            currentContextId = contextId
            activeModelName = modelName
            isCurrentlyLoaded = true
            lastLoadTimeMs = System.currentTimeMillis() - loadStartTime

            @Suppress("UNCHECKED_CAST")
            val modelDetailsMap = startResult["model"] as? Map<String, Any>
            loadedModelDetails = modelDetailsMap

            val detectedArch = modelDetailsMap?.get("architecture")?.toString()
                ?: modelDetailsMap?.get("desc")?.toString()
            val detectedCtx = (modelDetailsMap?.get("n_ctx_train") as? Number)?.toInt() ?: effectiveContextLength
            val detectedParams = (modelDetailsMap?.get("n_params") as? Number)?.let { num ->
                val paramsDouble = num.toDouble() / 1_000_000_000.0
                if (paramsDouble >= 0.1) String.format("%.1fB", paramsDouble) else null
            }

            AppLogger.i(
                "LlamaCppEngineBridge",
                "Model '$modelName' loaded successfully into context $contextId in ${lastLoadTimeMs}ms"
            )

            Result.success(
                EngineStatus(
                    isLoaded = true,
                    modelName = modelName,
                    memoryFootprintMb = fileSizeMb,
                    statusMessage = "Loaded in ${lastLoadTimeMs}ms. Hardware threads: $effectiveThreads, Context: $detectedCtx",
                    architecture = detectedArch,
                    contextLength = detectedCtx,
                    parametersCount = detectedParams
                )
            )
        } catch (e: Throwable) {
            AppLogger.e("LlamaCppEngineBridge", "Exception loading model '$modelName': ${e.message}", e)
            currentContextId = null
            isCurrentlyLoaded = false
            Result.failure(
                PocketAIException.ModelLoadFailedException(
                    modelName,
                    e.message ?: "Native engine initialization failure"
                )
            )
        }
    }

    override suspend fun unload(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val ctxId = currentContextId
            if (ctxId != null) {
                AppLogger.i("LlamaCppEngineBridge", "Releasing native context ID: $ctxId")
                try {
                    llamaAndroidInstance?.releaseContext(ctxId)
                } catch (e: Throwable) {
                    AppLogger.w("LlamaCppEngineBridge", "Error during releaseContext($ctxId): ${e.message}")
                }
            }
            currentContextId = null
            activeModelName = null
            isCurrentlyLoaded = false
            loadedModelDetails = null
            currentTokenListener = null
            AppLogger.i("LlamaCppEngineBridge", "Model unloaded and native memory released.")
            Result.success(Unit)
        } catch (e: Throwable) {
            AppLogger.e("LlamaCppEngineBridge", "Error releasing native context: ${e.message}", e)
            currentContextId = null
            isCurrentlyLoaded = false
            Result.failure(e)
        }
    }

    override suspend fun generate(
        prompt: String,
        contextHistory: String,
        params: ModelInferenceParams
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val fullResponse = StringBuilder()
            streamGenerate(prompt, contextHistory, params).collect { token ->
                fullResponse.append(token)
            }
            Result.success(fullResponse.toString())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override fun streamGenerate(
        prompt: String,
        contextHistory: String,
        params: ModelInferenceParams
    ): Flow<String> = channelFlow {
        val contextId = currentContextId
        val modelName = activeModelName

        if (!isCurrentlyLoaded || contextId == null || modelName == null) {
            throw PocketAIException.ModelNotLoadedException(
                activeModelName ?: "No model loaded. Please load a local GGUF model in Settings."
            )
        }

        val startTime = System.currentTimeMillis()
        var firstTokenTimeMs = 0L
        val tokenCounter = AtomicInteger(0)

        // Register the active token forwarder for this completion stream
        currentTokenListener = { token ->
            if (token.isNotEmpty()) {
                if (tokenCounter.get() == 0) {
                    firstTokenTimeMs = System.currentTimeMillis() - startTime
                }
                tokenCounter.incrementAndGet()
                trySend(token)
            }
        }

        try {
            // Note: LlamaContext requires Double for float-based hyperparameters
            val completionParams = mutableMapOf<String, Any>(
                "prompt" to prompt,
                "temperature" to params.temperature.toDouble(),
                "top_p" to params.topP.toDouble(),
                "top_k" to params.topK,
                "n_predict" to params.maxTokens,
                "penalty_repeat" to params.repeatPenalty.toDouble(),
                "emit_partial_completion" to true,
                "stop" to ChatTemplateFormatter.GLOBAL_STOP_TOKENS
            )

            withContext(Dispatchers.IO) {
                getLlamaAndroid().launchCompletion(contextId, completionParams)
            }
        } catch (e: Throwable) {
            if (e is CancellationException) {
                AppLogger.i("LlamaCppEngineBridge", "Inference cancelled by user or coroutine scope.")
                try {
                    getLlamaAndroid().stopCompletion(contextId)
                } catch (_: Throwable) {}
            } else {
                AppLogger.e("LlamaCppEngineBridge", "Inference failure: ${e.message}", e)
                throw PocketAIException.InferenceGenerationException(
                    e.message ?: "Native llama.cpp generation failure"
                )
            }
        } finally {
            currentTokenListener = null
            val totalTimeMs = (System.currentTimeMillis() - startTime).coerceAtLeast(1)
            val tokens = tokenCounter.get()
            val tokPerSec = if (tokens > 0) (tokens.toDouble() / (totalTimeMs.toDouble() / 1000.0)) else 0.0

            latestGenerationDiagnostics = GenerationDiagnostics(
                loadTimeMs = lastLoadTimeMs,
                tokensGenerated = tokens,
                generationTimeMs = totalTimeMs,
                tokensPerSecond = tokPerSec,
                timeToFirstTokenMs = firstTokenTimeMs,
                activeModelName = modelName
            )

            AppLogger.i(
                "LlamaCppEngineBridge",
                "Completion ended: $tokens tokens in ${totalTimeMs}ms (${String.format("%.2f", tokPerSec)} tok/s)"
            )
        }
    }

    override fun stopGeneration() {
        val contextId = currentContextId ?: return
        AppLogger.i("LlamaCppEngineBridge", "Requesting stopCompletion for context $contextId")
        engineScope.launch {
            try {
                llamaAndroidInstance?.stopCompletion(contextId)
            } catch (e: Throwable) {
                AppLogger.w("LlamaCppEngineBridge", "Error halting inference: ${e.message}")
            }
        }
    }
}
