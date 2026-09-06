package com.pocketai.local.engines

import com.pocketai.core.logging.AppLogger
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class GgufMetadata(
    val isValidGguf: Boolean,
    val version: Int = 0,
    val tensorCount: Long = 0L,
    val metadataKvCount: Long = 0L,
    val architecture: String? = null,
    val contextLength: Int? = null,
    val quantization: String? = null
)

object GgufHeaderParser {

    private const val GGUF_MAGIC = 0x46554747 // "GGUF" in little-endian

    /**
     * Attempts to parse GGUF header metadata from an InputStream.
     * Reads the magic header and basic key-values without loading the full file.
     */
    fun parse(inputStream: InputStream): GgufMetadata {
        return try {
            val headerBytes = ByteArray(16)
            var totalRead = 0
            while (totalRead < 16) {
                val r = inputStream.read(headerBytes, totalRead, 16 - totalRead)
                if (r == -1) break
                totalRead += r
            }

            if (totalRead < 16) {
                return GgufMetadata(isValidGguf = false)
            }

            val buffer = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = buffer.int
            if (magic != GGUF_MAGIC) {
                return GgufMetadata(isValidGguf = false)
            }

            val version = buffer.int
            val tensorCount = buffer.long

            // Read metadata kv count
            val kvCountBytes = ByteArray(8)
            inputStream.read(kvCountBytes)
            val kvBuffer = ByteBuffer.wrap(kvCountBytes).order(ByteOrder.LITTLE_ENDIAN)
            val metadataKvCount = kvBuffer.long

            // Best-effort key scanner from the header region
            var detectedArchitecture: String? = null
            var detectedContextLength: Int? = null
            var detectedQuant: String? = null

            val scanBuffer = ByteArray(4096)
            val scanRead = inputStream.read(scanBuffer)
            if (scanRead > 0) {
                val scanStr = String(scanBuffer, 0, scanRead, Charsets.ISO_8859_1)
                
                // Inspect common architecture tokens
                when {
                    scanStr.contains("bonsai", ignoreCase = true) -> detectedArchitecture = "bonsai"
                    scanStr.contains("llama", ignoreCase = true) -> detectedArchitecture = "llama"
                    scanStr.contains("qwen2", ignoreCase = true) || scanStr.contains("qwen", ignoreCase = true) -> detectedArchitecture = "qwen2"
                    scanStr.contains("gemma", ignoreCase = true) -> detectedArchitecture = "gemma"
                    scanStr.contains("phi3", ignoreCase = true) || scanStr.contains("phi", ignoreCase = true) -> detectedArchitecture = "phi"
                    scanStr.contains("mistral", ignoreCase = true) -> detectedArchitecture = "mistral"
                    scanStr.contains("deepseek", ignoreCase = true) -> detectedArchitecture = "deepseek"
                }

                // Inspect common quantization tokens
                val quantRegex = Regex("""(Q4_K_M|Q4_K_S|Q5_K_M|Q5_K_S|Q8_0|Q4_0|Q4_1|IQ4_XS|IQ3_M|BF16|F16)""")
                val quantMatch = quantRegex.find(scanStr)
                if (quantMatch != null) {
                    detectedQuant = quantMatch.value
                }

                // Inspect context length hint
                if (scanStr.contains("context_length", ignoreCase = true)) {
                    detectedContextLength = 4096
                }
            }

            GgufMetadata(
                isValidGguf = true,
                version = version,
                tensorCount = tensorCount,
                metadataKvCount = metadataKvCount,
                architecture = detectedArchitecture ?: "transformer",
                contextLength = detectedContextLength ?: 2048,
                quantization = detectedQuant
            )
        } catch (e: Exception) {
            AppLogger.w("GgufHeaderParser", "Error parsing GGUF header: ${e.message}")
            GgufMetadata(isValidGguf = false)
        }
    }
}
