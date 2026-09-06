package com.pocketai.brain

enum class MessageRole {
    USER,
    ASSISTANT,
    SYSTEM
}

enum class ModelSourceType {
    LOCAL,
    CLOUD,
    SYSTEM
}

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val modelSource: ModelSourceType = ModelSourceType.LOCAL,
    val modelName: String? = null,
    val latencyMs: Long? = null,
    val isStreaming: Boolean = false
)
