package com.pocketai.local.engines

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
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
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * Production bridge for on-device GGUF inference powered by llama.cpp.
 * Loads models from reliable local filesystem storage with conservative
 * memory and CPU configurations to ensure stability on mobile hardware.
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

    private fun validateGgufHeader(file: File): Boolean {
        return try {
            file.inputStream().use { stream ->
                val magicBytes = ByteArray(4)
                var total = 0
                while (total < 4) {
                    val r = stream.read(magicBytes, total, 4 - total)
                    if (r == -1) return false
                    total += r
                }
                // GGUF magic is 0x46554747 in little-endian ('G', 'G', 'U', 'F')
                magicBytes[0] == 'G'.code.toByte() &&
                magicBytes[1] == 'G'.code.toByte() &&
                magicBytes[2] == 'U'.code.toByte() &&
                magicBytes[3] == 'F'.code.toByte()
            }
        } catch (e: Exception) {
            false
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        if (uri.scheme == "file") return uri.lastPathSegment
        return try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (idx != -1) cursor.getString(idx) else null
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun queryFileSize(uri: Uri): Long {
        if (uri.scheme == "file") {
            return uri.path?.let { File(it).length() } ?: 0L
        }
        return try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idx = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (idx != -1) cursor.getLong(idx) else 0L
                } else 0L
            } ?: 0L
        } catch (e: Exception) {
            0L
        }
    }

    private fun ensureLocalModelFile(uri: Uri, modelName: String): File {
        val modelsDir = File(context.filesDir, "models").apply {
            if (!exists()) mkdirs()
        }

        // If the URI is already a local file path inside storage
        if (uri.scheme == "file" && uri.path != null) {
            val existingFile = File(uri.path!!)
            if (existingFile.exists() && existingFile.length() > 0) {
                if (validateGgufHeader(existingFile)) {
                    AppLogger.i("LlamaCppEngineBridge", "GGUF_HEADER_VALID: GGUF magic verified for ${existingFile.name}")
                    AppLogger.i("LlamaCppEngineBridge", "MODEL_FILE_VALIDATED: Using existing local file: ${existingFile.absolutePath} (${existingFile.length()} bytes)")
                    return existingFile
                } else {
                    AppLogger.w("LlamaCppEngineBridge", "File at ${existingFile.absolutePath} failed GGUF magic verification")
                }
            }
        }

        // Determine deterministic file name for local storage
        val queryName = queryDisplayName(uri) ?: modelName
        val cleanName = if (queryName.endsWith(".gguf", ignoreCase = true)) queryName else "$queryName.gguf"
        val safeFileName = cleanName.replace(Regex("[^a-zA-Z0-9._-]"), "_")
        val targetFile = File(modelsDir, safeFileName)
        val expectedSize = queryFileSize(uri)

        // If target file already exists and is valid, reuse it to avoid duplicate copies
        if (targetFile.exists() && targetFile.length() > 0) {
            val matchesSize = expectedSize <= 0 || targetFile.length() == expectedSize
            if (matchesSize && validateGgufHeader(targetFile)) {
                AppLogger.i("LlamaCppEngineBridge", "GGUF_HEADER_VALID: GGUF magic verified for ${targetFile.name}")
                AppLogger.i("LlamaCppEngineBridge", "MODEL_FILE_VALIDATED: Deterministic local model file already exists: ${targetFile.absolutePath} (${targetFile.length()} bytes)")
                return targetFile
            }
        }

        // Copy bytes from ContentResolver to app-private storage
        AppLogger.i("LlamaCppEngineBridge", "LOCAL_MODEL_COPY_START: Copying $uri to ${targetFile.absolutePath}")
        val tempFile = File(modelsDir, "${safeFileName}.tmp.${System.currentTimeMillis()}")
        try {
            val inputStream = context.contentResolver.openInputStream(uri)
                ?: throw PocketAIException.ModelNotFoundException("Failed to open input stream for $uri")

            val buffer = ByteArray(64 * 1024)
            var totalCopied = 0L
            tempFile.outputStream().use { outputStream ->
                inputStream.use { stream ->
                    var bytesRead: Int
                    while (stream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                        totalCopied += bytesRead
                    }
                    outputStream.flush()
                }
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }
            val renamed = tempFile.renameTo(targetFile)
            if (!renamed) {
                tempFile.copyTo(targetFile, overwrite = true)
                tempFile.delete()
            }

            AppLogger.i("LlamaCppEngineBridge", "LOCAL_MODEL_COPY_SUCCESS: Copied $totalCopied bytes to ${targetFile.absolutePath}")
        } catch (e: Exception) {
            if (tempFile.exists()) tempFile.delete()
            AppLogger.e("LlamaCppEngineBridge", "Failed during local model copy: ${e.message}", e)
            throw e
        }

        // Verify copied file existence and size
        if (!targetFile.exists() || targetFile.length() == 0L) {
            throw PocketAIException.ModelNotFoundException("Copied local model file does not exist or is empty: ${targetFile.absolutePath}")
        }

        if (expectedSize > 0 && targetFile.length() != expectedSize) {
            AppLogger.w("LlamaCppEngineBridge", "Warning: Copied file size (${targetFile.length()}) differs from SAF reported size ($expectedSize)")
        }

        // Verify GGUF header
        if (!validateGgufHeader(targetFile)) {
            AppLogger.e("LlamaCppEngineBridge", "GGUF header validation failed on copied file: ${targetFile.absolutePath}")
            targetFile.delete()
            throw PocketAIException.ModelLoadFailedException(modelName, "Copied file is not in valid GGUF format")
        }

        AppLogger.i("LlamaCppEngineBridge", "GGUF_HEADER_VALID: GGUF magic verified for ${targetFile.name}")
        AppLogger.i("LlamaCppEngineBridge", "MODEL_FILE_VALIDATED: Verified local model file at ${targetFile.absolutePath} (${targetFile.length()} bytes)")
        return targetFile
    }

    override suspend fun loadModelFromUri(
        uri: Uri,
        modelName: String,
        params: ModelInferenceParams
    ): Result<EngineStatus> = withContext(Dispatchers.IO) {
        AppLogger.i("LlamaCppEngineBridge", "MODEL_LOAD_START: modelName='$modelName', uri='$uri'")
        AppLogger.i("LlamaCppEngineBridge", "SAF_URI_RECEIVED: $uri")

        try {
            if (!isNativeEngineAvailable) {
                AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: llama.cpp native binary is not available on this device runtime.")
                return@withContext Result.failure(
                    PocketAIException.NativeEngineUnavailableException(
                        "llama.cpp native binary is not available on this device runtime."
                    )
                )
            }

            val localFile = try {
                ensureLocalModelFile(uri, modelName)
            } catch (e: Exception) {
                AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: Failed to ensure local model file: ${e.message}", e)
                return@withContext Result.failure(e)
            }

            val fileSizeBytes = localFile.length()
            val fileSizeMb = fileSizeBytes / (1024 * 1024)
            val diagnostics = getHardwareDiagnostics()

            // Conservative parameters for Android CPU stability
            val effectiveContextLength = if (params.contextLength > 0) params.contextLength else 2048
            val effectiveThreads = if (params.threads > 0) params.threads.coerceIn(2, 4) else diagnostics.processorCores.coerceIn(2, 4)

            // Safe memory calculation: model weights + KV cache headroom (approx 250MB for 2048 context)
            val estimatedKvCacheMb = (effectiveContextLength * 4) / 1024 + 100
            val requiredMemoryMb = fileSizeMb + estimatedKvCacheMb

            if (diagnostics.availableRamMb < requiredMemoryMb) {
                val errorMsg = "Insufficient RAM for '$modelName'. Required: ~${requiredMemoryMb}MB (Model: ${fileSizeMb}MB + Context: ${estimatedKvCacheMb}MB), Available: ${diagnostics.availableRamMb}MB"
                AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: $errorMsg")
                return@withContext Result.failure(
                    PocketAIException.InsufficientMemoryException(
                        requiredMb = requiredMemoryMb,
                        availableMb = diagnostics.availableRamMb
                    )
                )
            }

            // If another model is currently loaded in memory, release it first
            if (isCurrentlyLoaded) {
                unload()
            }

            val loadStartTime = System.currentTimeMillis()
            AppLogger.i("LlamaCppEngineBridge", "NATIVE_ENGINE_START: Initializing llama.cpp engine for '${localFile.name}'")

            val pfd = try {
                ParcelFileDescriptor.open(localFile, ParcelFileDescriptor.MODE_READ_ONLY)
            } catch (e: Exception) {
                AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: Cannot open ParcelFileDescriptor: ${e.message}", e)
                return@withContext Result.failure(
                    PocketAIException.StoragePermissionException("Cannot open file descriptor for ${localFile.absolutePath}: ${e.message}")
                )
            }

            // Detach raw file descriptor: native initContextWithFd takes ownership of this FD, dups it, and closes it
            val fd = pfd.detachFd()
            var nativeTookOwnership = false

            try {
                val fileUriString = Uri.fromFile(localFile).toString()

                val startParams = mutableMapOf<String, Any>(
                    "model" to fileUriString,
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
                // Limit simultaneous active contexts to 1 for mobile memory safety
                try {
                    engine.setContextLimit(1)
                } catch (_: Throwable) {
                    // Ignore if setContextLimit is not supported
                }

                val tokenForwarder: (String) -> Unit = { token ->
                    currentTokenListener?.invoke(token)
                }

                AppLogger.i("LlamaCppEngineBridge", "CONTEXT_CREATE_START: Calling startEngine for '${localFile.name}' (ctx: $effectiveContextLength, threads: $effectiveThreads, n_gpu_layers: 0)")

                val startResult = engine.startEngine(startParams, tokenForwarder)
                nativeTookOwnership = true

                if (startResult == null || !startResult.containsKey("contextId")) {
                    AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: startEngine returned null or missing contextId")
                    return@withContext Result.failure(
                        PocketAIException.ModelLoadFailedException(
                            modelName,
                            "Native llama.cpp engine failed to initialize context from GGUF weights."
                        )
                    )
                }

                val contextId = (startResult["contextId"] as? Number)?.toInt()
                if (contextId == null || contextId <= 0) {
                    AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: Engine returned an invalid native context identifier: $contextId")
                    return@withContext Result.failure(
                        PocketAIException.ModelLoadFailedException(
                            modelName,
                            "Engine returned an invalid native context identifier: $contextId"
                        )
                    )
                }

                AppLogger.i("LlamaCppEngineBridge", "CONTEXT_CREATE_SUCCESS: Created native context ID: $contextId")

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
                    "MODEL_LOAD_SUCCESS: Model '$modelName' loaded successfully into context $contextId in ${lastLoadTimeMs}ms (Arch: $detectedArch, Ctx: $detectedCtx, Params: $detectedParams)"
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
                if (!nativeTookOwnership) {
                    try {
                        ParcelFileDescriptor.adoptFd(fd).close()
                    } catch (_: Throwable) {}
                }
                AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: Exception loading model '$modelName': ${e.message}", e)
                currentContextId = null
                isCurrentlyLoaded = false
                Result.failure(
                    PocketAIException.ModelLoadFailedException(
                        modelName,
                        e.message ?: "Native engine initialization failure"
                    )
                )
            }
        } catch (e: Throwable) {
            AppLogger.e("LlamaCppEngineBridge", "MODEL_LOAD_FAILED: Unexpected error loading model '$modelName': ${e.message}", e)
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
                AppLogger.i("LlamaCppEngineBridge", "MODEL_UNLOAD_START: Releasing native context ID: $ctxId")
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
            AppLogger.i("LlamaCppEngineBridge", "MODEL_UNLOAD_SUCCESS: Model unloaded and native memory released.")
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

