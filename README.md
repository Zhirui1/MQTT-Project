## Overview

This repository open-sources the **testing code**, **MQTT communication code**, and the **web UI (frontend + optional backend)**.

**Not included**: any model training code, model weights/checkpoints, large datasets, or environment-specific/private configuration.

## Visualization

Example visualization (from `visual-PreMixHuge/299.jpg`):

![Visualization example](visual-PreMixHuge/299.jpg)

## Features

- **MQTT client/server demo** for message publishing/subscribing
- **Web UI (Vite + React)** for interacting with the system
- **Backend (Python)** to support the web UI (if needed by your setup)
- **Basic tests** to validate core behaviors

## Project Layout

```
.
├── MQTT_PANSHARPENING_README.md
├── mqtt_pansharpening_server.py
├── mqtt_pansharpening_client.py
├── local_client/
│   └── mqtt_client.py
├── web_ui/
│   ├── backend/
│   │   ├── app.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       ├── package.json
│       ├── package-lock.json
│       └── vite.config.js
└── test.py
```

## Requirements

- **Python**: 3.8+ (recommended)
- **Node.js**: 18+ (recommended)
- **MQTT Broker**: Mosquitto / EMQX / etc.

## Quick Start

### 1) (Optional) Start an MQTT broker

If you already have one, skip this step.

Make sure you know:

- Broker host: `YOUR_BROKER_HOST`
- Broker port: `YOUR_BROKER_PORT`
- Username/password (if enabled)

### 2) Run the backend (optional)

If your web UI requires the backend, start it first:

```bash
cd web_ui/backend
pip install -r requirements.txt
python app.py
```

### 3) Run the frontend

```bash
cd web_ui/frontend
npm install
npm run dev
```

Then open the URL printed by Vite (commonly `http://localhost:5173`).

### 4) Run MQTT demo scripts

In two terminals:

```bash
python mqtt_pansharpening_server.py
```

```bash
python mqtt_pansharpening_client.py
```

## Configuration

MQTT connection parameters (broker host/port/auth/topics) are typically configured inside:

- `mqtt_pansharpening_server.py`
- `mqtt_pansharpening_client.py`
- `local_client/mqtt_client.py`

If you plan to publish this repo, **do not hardcode secrets** (passwords/tokens). Prefer environment variables or a local config file that is ignored by git.

## Tests

```bash
python test.py
```

## Open-source Scope (What’s Included / Excluded)

### Included

- MQTT scripts and web UI code
- Test code
- Documentation and configuration needed to run the demo

### Excluded

- Training code, training pipelines, and training datasets
- Model weights/checkpoints and artifacts (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.h5`, `*.pkl`, etc.)
- Local IDE / deployment configuration (e.g. `.idea/`)
- Frontend dependencies and build outputs (`node_modules/`, `dist/`, `build/`)
- Any private keys, tokens, `.env` files, or credentials

## Security Notes

Before publishing:

- Remove any files that contain IPs, usernames, ports, or credentials.
- Do **not** commit IDE deployment configs. For example, JetBrains `.idea/deployment.xml` may contain server connection details.

## License

MIT License (see `LICENSE`).

