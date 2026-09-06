package com.pocketai.ui.models

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pocketai.local.LocalModelInfo
import com.pocketai.ui.components.EmptyStateView
import com.pocketai.ui.components.StatusBadge
import com.pocketai.ui.settings.SettingsViewModel
import com.pocketai.ui.theme.AmberWarning
import com.pocketai.ui.theme.CoralError
import com.pocketai.ui.theme.CyberCyan
import com.pocketai.ui.theme.EmeraldSuccess

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocalModelsScreen(
    viewModel: SettingsViewModel,
    onNavigateBack: () -> Unit
) {
    val localModels by viewModel.localModels.collectAsState()
    val loadedModel by viewModel.loadedLocalModel.collectAsState()
    val operationMessage by viewModel.operationMessage.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }
    val diagnostics = remember { viewModel.getHardwareDiagnostics() }

    // SAF Document Picker for .gguf files
    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.importLocalModel(it) }
    }

    LaunchedEffect(operationMessage) {
        operationMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.dismissOperationMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Local GGUF Models", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(
                        onClick = onNavigateBack,
                        modifier = Modifier.testTag("local_models_back_button")
                    ) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(
                        onClick = { filePicker.launch(arrayOf("*/*")) },
                        modifier = Modifier.testTag("import_model_icon_button")
                    ) {
                        Icon(Icons.Default.Add, contentDescription = "Import Model")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface)
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) },
        contentWindowInsets = WindowInsets(0, 0, 0, 0)
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Hardware Diagnostics Card
            item {
                Spacer(modifier = Modifier.height(4.dp))
                HardwareDiagnosticsCard(
                    totalRamMb = diagnostics.totalRamMb,
                    availableRamMb = diagnostics.availableRamMb,
                    cores = diagnostics.processorCores,
                    maxSafeMb = diagnostics.recommendedMaxModelSizeMb
                )
            }

            // Import CTA Banner
            item {
                Button(
                    onClick = { filePicker.launch(arrayOf("*/*")) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("import_model_button"),
                    colors = ButtonDefaults.buttonColors(containerColor = CyberCyan),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.FileDownload, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Import Model File (.gguf)", fontWeight = FontWeight.Bold)
                }
            }

            if (localModels.isEmpty()) {
                item {
                    EmptyStateView(
                        icon = Icons.Default.Memory,
                        title = "No Local Models Imported",
                        description = "Download a GGUF model (e.g. bonsai-1.7b.gguf, llama-3.2-1b.gguf, qwen2.5-0.5b.gguf) to your phone storage, then tap 'Import Model File' above to load it into PocketAI.",
                        actionLabel = "Select GGUF File",
                        onAction = { filePicker.launch(arrayOf("*/*")) }
                    )
                }
            } else {
                item {
                    Text(
                        text = "IMPORTED MODELS (${localModels.size})",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold, letterSpacing = 1.sp),
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                items(localModels) { model ->
                    val isCurrentLoaded = loadedModel?.id == model.id
                    LocalModelCard(
                        model = model,
                        isLoaded = isCurrentLoaded,
                        onLoad = { viewModel.loadLocalModel(model.id) },
                        onUnload = { viewModel.unloadLocalModel(model.id) },
                        onDelete = { viewModel.deleteLocalModel(model.id) }
                    )
                }
            }

            item {
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
fun HardwareDiagnosticsCard(
    totalRamMb: Long,
    availableRamMb: Long,
    cores: Int,
    maxSafeMb: Long
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Hardware Headroom",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                StatusBadge(
                    text = if (availableRamMb > 1500) "OPTIMAL" else "CONSTRAINED",
                    color = if (availableRamMb > 1500) EmeraldSuccess else AmberWarning
                )
            }
            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricColumn(label = "Free RAM", value = "${availableRamMb}MB")
                MetricColumn(label = "Total RAM", value = "${totalRamMb}MB")
                MetricColumn(label = "CPU Cores", value = "$cores Cores")
                MetricColumn(label = "Rec. Max GGUF", value = "${maxSafeMb}MB")
            }
        }
    }
}

@Composable
fun MetricColumn(label: String, value: String) {
    Column {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(text = value, style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold), color = MaterialTheme.colorScheme.onSurface)
    }
}

@Composable
fun LocalModelCard(
    model: LocalModelInfo,
    isLoaded: Boolean,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isLoaded) CyberCyan.copy(alpha = 0.08f) else MaterialTheme.colorScheme.surface
        ),
        border = androidx.compose.foundation.BorderStroke(
            1.dp,
            if (isLoaded) CyberCyan else MaterialTheme.colorScheme.outline
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(if (isLoaded) EmeraldSuccess.copy(alpha = 0.2f) else MaterialTheme.colorScheme.surfaceVariant),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = if (isLoaded) Icons.Default.CheckCircle else Icons.Default.Memory,
                            contentDescription = null,
                            tint = if (isLoaded) EmeraldSuccess else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = model.name,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = model.fileName,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                StatusBadge(
                    text = if (isLoaded) "LOADED" else "UNLOADED",
                    color = if (isLoaded) EmeraldSuccess else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Spec chips
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val sizeMb = model.sizeBytes / (1024 * 1024)
                val sizeText = if (sizeMb > 1024) String.format("%.2f GB", sizeMb / 1024.0) else "$sizeMb MB"
                StatusBadge(text = sizeText, color = CyberCyan)
                StatusBadge(text = model.format, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (!model.architecture.isNullOrBlank()) {
                    StatusBadge(text = model.architecture, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (!model.quantization.isNullOrBlank()) {
                    StatusBadge(text = model.quantization, color = AmberWarning)
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = onDelete,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.DeleteOutline,
                        contentDescription = "Delete Model",
                        tint = CoralError
                    )
                }

                if (isLoaded) {
                    OutlinedButton(
                        onClick = onUnload,
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = AmberWarning),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Unload from RAM")
                    }
                } else {
                    Button(
                        onClick = onLoad,
                        colors = ButtonDefaults.buttonColors(containerColor = EmeraldSuccess),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Load Model", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }
    }
}
