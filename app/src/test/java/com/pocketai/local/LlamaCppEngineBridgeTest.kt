package com.pocketai.local

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.pocketai.local.engines.LlamaCppEngineBridge
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

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
    fun testStopGenerationWhenIdleDoesNotCrash() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val bridge = LlamaCppEngineBridge(context)
        // Calling stopGeneration when no generation is ongoing should be completely safe
        bridge.stopGeneration()
    }
}
