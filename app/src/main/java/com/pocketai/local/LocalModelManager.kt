package com.pocketai.local

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import com.pocketai.brain.ChatMessage
import com.pocketai.core.error.PocketAIException
import com.pocketai.core.logging.AppLogger
import com.pocketai.local.engines.GgufHeaderParser
import com.pocketai.local.engines.InferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.util.UUID

class LocalModelManager(
    private val context: Context,
    private val repository: LocalModelRepository,
    val engine: InferenceEngine
) : LocalModelProvider {

    suspend fun importModelFromUri(uri: Uri): Result<LocalModelInfo> = withContext(Dispatchers.IO) {
        try {
            // Persist read permissions for Storage Access Framework URI
            try {
                val takeFlags: Int = Intent.FLAG_GRANT_READ_URI_PERMISSION
                context.contentResolver.takePersistableUriPermission(uri, takeFlags)
            } catch (e: Exception) {
                AppLogger.w("LocalModelManager", "Could not take persistable URI permission: ${e.message}")
            }

            var displayName = "model.gguf"
            var fileSize = 0L

            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                    if (nameIdx != -1) displayName = cursor.getString(nameIdx) ?: displayName
                    if (sizeIdx != -1) fileSize = cursor.getLong(sizeIdx)
                }
            }

            // Inspect GGUF header
            val metadata = context.contentResolver.openInputStream(uri)?.use { stream ->
                GgufHeaderParser.parse(stream)
            }

            val cleanModelName = displayName.removeSuffix(".gguf")
                .replace("-", " ")
                .replace("_", " ")
                .split(" ")
                .joinToString(" ") { it.replaceFirstChar { char -> char.uppercase() } }

            val modelInfo = LocalModelInfo(
                id = UUID.randomUUID().toString(),
                name = cleanModelName,
                fileName = displayName,
                fileUri = uri.toString(),
                format = if (metadata?.isValidGguf == true) "GGUF v${metadata.version}" else "GGUF",
                sizeBytes = fileSize,
                architecture = metadata?.architecture ?: "transformer",
                quantization = metadata?.quantization,
                contextLength = metadata?.contextLength ?: 2048,
                status = LocalModelStatus.UNLOADED,
                isLoaded = false,
                importTimestamp = System.currentTimeMillis()
            )

            repository.saveModel(modelInfo)
            AppLogger.i("LocalModelManager", "Imported model '$cleanModelName' (${fileSize / (1024 * 1024)}MB)")
            Result.success(modelInfo)
        } catch (e: Exception) {
            AppLogger.e("LocalModelManager", "Failed to import model from URI: ${e.message}", e)
            Result.failure(e)
        }
    }

    override suspend fun loadModel(modelId: String): Result<Unit> = withContext(Dispatchers.IO) {
        val model = repository.getModelById(modelId)
            ?: return@withContext Result.failure(PocketAIException.ModelNotFoundException(modelId))

        AppLogger.i("LocalModelManager", "Loading model '${model.name}'")
        val uri = Uri.parse(model.fileUri)

        val loadResult = engine.loadModelFromUri(uri, model.name)
        if (loadResult.isSuccess) {
            repository.markLoaded(modelId)
            Result.success(Unit)
        } else {
            val error = loadResult.exceptionOrNull() ?: Exception("Failed to load model")
            Result.failure(error)
        }
    }

    override suspend fun unloadModel(modelId: String): Result<Unit> = withContext(Dispatchers.IO) {
        engine.unload()
        repository.markAllUnloaded()
        AppLogger.i("LocalModelManager", "Model $modelId unloaded.")
        Result.success(Unit)
    }

    suspend fun deleteModel(modelId: String): Result<Unit> = withContext(Dispatchers.IO) {
        val model = repository.getModelById(modelId)
        if (model?.isLoaded == true) {
            unloadModel(modelId)
        }
        repository.deleteModel(modelId)
        AppLogger.i("LocalModelManager", "Model $modelId deleted from registry.")
        Result.success(Unit)
    }

    override suspend fun generate(prompt: String, context: List<ChatMessage>): Result<String> {
        val contextHistory = context.joinToString("\n") { "${it.role}: ${it.content}" }
        return engine.generate(prompt, contextHistory)
    }

    override fun streamGenerate(prompt: String, context: List<ChatMessage>): Flow<String> {
        val contextHistory = context.joinToString("\n") { "${it.role}: ${it.content}" }
        return engine.streamGenerate(prompt, contextHistory)
    }

    override fun getModelInfo(modelId: String): LocalModelInfo? {
        return null // Will be queried via repository asynchronously
    }

    override fun isLoaded(): Boolean {
        return false // Reactive state is tracked via repository.loadedModel
    }

    override fun releaseResources() {
        // Cleanup if needed
    }
}
