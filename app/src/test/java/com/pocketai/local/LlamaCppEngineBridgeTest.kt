package com.pocketai.local

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.pocketai.local.engines.GgufHeaderParser
import com.pocketai.local.engines.LlamaCppEngineBridge
import com.pocketai.local.engines.ModelInferenceParams
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class LlamaCppEngineBridgeTest {

    @Test
    fun testLlamaCppEngineBridgeInitialization() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)

        assertEquals("llama.cpp GGUF Engine", bridge.engineName)
        assertFalse("Bridge should not report loaded initially", bridge.isLoaded())
    }

    @Test
    fun testHardwareDiagnostics() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)
        val diagnostics = bridge.getHardwareDiagnostics()

        assertTrue("Total RAM should be positive", diagnostics.totalRamMb > 0)
        assertTrue("Available RAM should be positive", diagnostics.availableRamMb > 0)
        assertTrue("Cores count should be positive", diagnostics.processorCores > 0)
        assertTrue("Recommended model size should be at least 400MB", diagnostics.recommendedMaxModelSizeMb >= 400)
    }

    @Test
    fun testLlamaCppClassesExistOnClasspath() {
        val classLoader = this.javaClass.classLoader
        val llamaAndroidClass = Class.forName("org.nehuatl.llamacpp.LlamaAndroid", false, classLoader)
        val llamaContextClass = Class.forName("org.nehuatl.llamacpp.LlamaContext", false, classLoader)
        val llamaHelperClass = Class.forName("org.nehuatl.llamacpp.LlamaHelper", false, classLoader)

        assertNotNull(llamaAndroidClass)
        assertNotNull(llamaContextClass)
        assertNotNull(llamaHelperClass)

        // Verify key methods exist on LlamaAndroid
        val startEngineMethod = llamaAndroidClass.methods.find { it.name == "startEngine" }
        val launchCompletionMethod = llamaAndroidClass.methods.find { it.name == "launchCompletion" }
        val releaseContextMethod = llamaAndroidClass.methods.find { it.name == "releaseContext" }
        val stopCompletionMethod = llamaAndroidClass.methods.find { it.name == "stopCompletion" }

        assertNotNull("startEngine method must exist on LlamaAndroid", startEngineMethod)
        assertNotNull("launchCompletion method must exist on LlamaAndroid", launchCompletionMethod)
        assertNotNull("releaseContext method must exist on LlamaAndroid", releaseContextMethod)
        assertNotNull("stopCompletion method must exist on LlamaAndroid", stopCompletionMethod)
    }

    @Test
    fun testGgufHeaderParserValidGemma2bQ2K() {
        val buffer = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(0x46554747) // "GGUF"
        buffer.putInt(3)          // version 3
        buffer.putLong(100L)      // tensor count
        buffer.putLong(10L)       // kv count
        buffer.put("gemma.context_length".toByteArray(Charsets.ISO_8859_1))

        val stream = ByteArrayInputStream(buffer.array())
        val metadata = GgufHeaderParser.parse(stream, "gemma-2b-it-Q2_K.gguf")

        assertTrue("Should be valid GGUF", metadata.isValidGguf)
        assertEquals("gemma", metadata.architecture)
        assertEquals("Q2_K", metadata.quantization)
    }

    @Test
    fun testGgufHeaderParserValidBonsai17bQ1_0() {
        val buffer = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(0x46554747) // "GGUF"
        buffer.putInt(3)          // version 3
        buffer.putLong(80L)       // tensor count
        buffer.putLong(12L)       // kv count
        buffer.put("bonsai.context_length".toByteArray(Charsets.ISO_8859_1))

        val stream = ByteArrayInputStream(buffer.array())
        val metadata = GgufHeaderParser.parse(stream, "Bonsai-1.7B-Q1_0.gguf")

        assertTrue("Should be valid GGUF", metadata.isValidGguf)
        assertEquals("bonsai", metadata.architecture)
        assertEquals("Q1_0", metadata.quantization)
    }

    @Test
    fun testGgufHeaderParserRejectsNonGguf() {
        val nonGgufBytes = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0x04, 0x05)
        val stream = ByteArrayInputStream(nonGgufBytes)
        val metadata = GgufHeaderParser.parse(stream)

        assertFalse("Non-GGUF bytes should be rejected", metadata.isValidGguf)
    }

    @Test
    fun testLoadNonExistentModelFailsGracefully() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)
        val nonExistentUri = android.net.Uri.fromFile(File(context.filesDir, "non_existent.gguf"))

        val result = bridge.loadModelFromUri(
            nonExistentUri,
            "NonExistent",
            ModelInferenceParams()
        )

        assertTrue("Loading non-existent file should return failure", result.isFailure)
        assertFalse("Bridge should not report loaded", bridge.isLoaded())
    }

    @Test
    fun testUnloadWhenIdleSucceeds() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)
        val unloadResult = bridge.unload()

        assertTrue("Unload when idle should succeed", unloadResult.isSuccess)
        assertFalse("Bridge should remain unloaded", bridge.isLoaded())
    }

    @Test
    fun testStopGenerationWhenIdleDoesNotCrash() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)
        // Calling stopGeneration when no generation is ongoing should be completely safe
        bridge.stopGeneration()
    }
}
