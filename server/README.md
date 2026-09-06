# PocketAI Server Gateway (Future Architecture)

This directory is reserved for the future PocketAI Server gateway component.

## Role in Architecture
```
Phone (PocketAI Android App)
       ↓
PocketAI Server (Secure Gateway)
       ↓
ESP32 Robot Controller
```

## Key Security Mandate
- Cloud provider API keys are **NEVER** stored directly inside the ESP32 microcontroller firmware.
- The PocketAI Server acts as an authenticated intermediary, issuing ephemeral session tokens to robots and managing cloud/local inference multiplexing.
