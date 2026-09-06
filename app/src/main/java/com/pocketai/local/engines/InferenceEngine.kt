package com.pocketai.local.engines

import android.net.Uri
import kotlinx.coroutines.flow.Flow

interface InferenceEngine {
    val engineName: String
    val isNativeEngineAvailable: Boolean

    suspend fun loadModelFromUri(uri: Uri, modelName: String): Result<EngineStatus>
    suspend fun unload(): Result<Unit>
    suspend fun generate(prompt: String, contextHistory: String): Result<String>
    fun streamGenerate(prompt: String, contextHistory: String): Flow<String>
    fun getHardwareDiagnostics(): HardwareDiagnostics
}

data class EngineStatus(
    val isLoaded: Boolean,
    val modelName: String,
    val memoryFootprintMb: Long,
    val statusMessage: String
)

data class HardwareDiagnostics(
    val totalRamMb: Long,
    val availableRamMb: Long,
    val isHardwareSufficient: Boolean,
    val recommendedMaxModelSizeMb: Long,
    val processorCores: Int
)
