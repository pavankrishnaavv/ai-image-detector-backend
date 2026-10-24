# ⚡ AI Image Detector Backend Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/umm-maybe/AI-image-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance, production-ready backend engine and REST API to detect whether an image is **AI-generated** or **human-created**. Powered by the Hugging Face [**`umm-maybe/AI-image-detector`**](https://huggingface.co/umm-maybe/AI-image-detector) Swin Transformer architecture.

This engine includes:
- 🚀 **High-Speed Inference**: Sub-250ms CPU latency (sub-30ms with CUDA GPU acceleration).
- 📦 **Self-Contained & Offline-Ready**: Bundled with pre-exported local model weights (`./model_weights`) so it runs without external HF network dependencies.
- 🌐 **Flexible Input Modes**: Multipart file upload, public image URLs, raw base64 data URIs, and batch payloads.
- 🖥️ **Interactive Built-in Web UI**: Test images instantly via drag-and-drop or URL directly from your browser.
- 🛠️ **CLI Tooling**: Run classifications directly from the terminal.
- 🐳 **Docker & Production Ready**: Pre-configured Dockerfile, Docker Compose, CORS middleware, and health telemetry.

---

## 📑 Table of Contents
1. [Model Overview](#-model-overview)
2. [Architecture & Project Structure](#-architecture--project-structure)
3. [Quickstart & Installation](#-quickstart--installation)
4. [Interactive Testing Dashboard](#-interactive-testing-dashboard)
5. [CLI Tool Usage](#-cli-tool-usage)
6. [REST API Documentation](#-rest-api-documentation)
7. [Website Integration Guide](#-website-integration-guide)
   - [Vanilla JavaScript / Fetch API](#1-vanilla-javascript--fetch-api)
   - [React / Next.js Component](#2-react--nextjs-integration)
   - [Vue.js 3 Component](#3-vuejs-3-integration)
   - [Node.js / Express Backend Proxy](#4-nodejs--express-backend-proxy)
   - [Python Client](#5-python-client)
8. [Extensive & Production Use Guide](#-extensive--production-use-guide)
   - [Multi-Worker Concurrency](#1-multi-worker-concurrency-gunicorn--uvicorn)
   - [GPU Acceleration (CUDA)](#2-gpu-acceleration)
   - [High-Scale Queue Worker (Celery + Redis)](#3-high-scale-batch--queue-worker-architecture)
   - [Rate Limiting & Authentication](#4-rate-limiting--api-security)
   - [Decision Threshold Calibration](#5-decision-threshold-calibration)
   - [Model Capabilities & Limitations](#6-model-capabilities--limitations)
9. [Running Automated Tests](#-running-automated-tests)
10. [License](#-license)

---

## 🧠 Model Overview

- **Model Identifier**: [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector)
- **Base Architecture**: `SwinForImageClassification` (Shifted Window Vision Transformer, Swin-Base, 224x224 patch resolution).
- **Classification Head**: Binary classification (`0: artificial`, `1: human`).
- **Input Preprocessing**: Image normalization, resizing to 224x224 px, center cropping, and standard ImageNet channel scaling.
- **Intended Scope**: Specialized in detecting generative art artifacts (e.g., Stable Diffusion, Midjourney, DALL-E, VQGAN).

---

## 🏗️ Architecture & Project Structure

```text
ai-image-detector-backend/
├── engine/
│   ├── __init__.py          # Package initialization
│   └── detector.py          # Core Swin Transformer inference engine
├── model_weights/           # Bundled model weights & preprocessor configs
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors    # Tracked with Git LFS
├── static/
│   └── index.html           # Interactive web dashboard
├── tests/
│   ├── __init__.py
│   ├── test_engine.py       # Engine unit tests
│   ├── test_api.py          # FastAPI endpoint integration tests
│   ├── generate_test_images.py
│   └── sample_images/       # Sample test fixtures
├── cli.py                   # Terminal testing tool
├── main.py                  # FastAPI application with REST endpoints
├── save_model.py            # Utility script to export/cache weights
├── requirements.txt         # Python package dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose setup
├── .gitattributes           # Git LFS rules for large model weights
└── README.md                # Comprehensive documentation
```

---

## 🚀 Quickstart & Installation

### Option 1: Run Locally (Python 3.10+)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/ai-image-detector-backend.git
   cd ai-image-detector-backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API Server**:
   ```bash
   python main.py
   # Or using uvicorn directly:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Open your browser:
   - **Interactive Web UI**: [http://localhost:8000/](http://localhost:8000/)
   - **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

### Option 2: Run with Docker

```bash
# Build and run container
docker compose up --build -d

# Verify logs
docker compose logs -f
```

The service will be live on `http://localhost:8000`.

---

## 🖥️ Interactive Testing Dashboard

When you open `http://localhost:8000/` in your browser, the backend serves an embedded telemetry dashboard:
- **Drag-and-Drop Dropzone**: Drag any PNG, JPG, WebP file onto the dropzone.
- **URL Analyzer**: Paste any public image link to test remote downloads.
- **Dynamic Threshold Slider**: Adjust the AI decision threshold (0.10 to 0.90) in real-time.
- **Dual Confidence Gauge**: Visualizes exact percentage probabilities for Artificial vs Human.
- **Raw JSON Inspector**: View and copy the exact JSON payload returned to integrating clients.

---

## 💻 CLI Tool Usage

Test local images or URLs without starting an HTTP server:

```bash
# Test a local image file
python cli.py path/to/sample.jpg

# Test a remote image URL
python cli.py --url "https://images.unsplash.com/photo-1546768292-fb12f6c92568"

# Custom sensitivity threshold
python cli.py path/to/sample.jpg --threshold 0.65

# Batch analysis across multiple images
python cli.py --batch art1.png art2.jpg photo1.webp

# Output raw JSON to pipe into jq or scripts
python cli.py path/to/sample.jpg --json
```

**CLI Output Example**:
```text
==================================================
Verdict:            [AI-GENERATED] AI-Generated
Confidence:         81.15%
Artificial Score:   81.15%
Human Score:        18.85%
Decision Threshold: 0.5
Image Resolution:   1536x674 px
Inference Latency:  224.39 ms
Device:             CPU
==================================================
```

---

## 📡 REST API Documentation

### 1. `POST /api/v1/detect`
Upload a single image file via `multipart/form-data`.

**Parameters**:
- `file` (form-data, required): The binary image file (`image/jpeg`, `image/png`, `image/webp`, etc.).
- `threshold` (query param, optional, default `0.5`): Floating point number between `0.0` and `1.0`.

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect?threshold=0.5" \
  -H "accept: application/json" \
  -F "file=@/path/to/my_image.png"
```

**Response (`200 OK`)**:
```json
{
  "success": true,
  "is_ai_generated": true,
  "verdict": "AI-Generated",
  "confidence": 0.8115,
  "confidence_percentage": "81.15%",
  "scores": {
    "artificial": 0.8115,
    "human": 0.1885
  },
  "threshold": 0.5,
  "image_size": {
    "width": 1536,
    "height": 674
  },
  "inference_time_ms": 224.39,
  "model": "umm-maybe/AI-image-detector",
  "device": "cpu"
}
```

---

### 2. `POST /api/v1/detect/url`
Detect from a publicly accessible HTTP/HTTPS image URL.

**Request Body**:
```json
{
  "url": "https://example.com/artwork.jpg",
  "threshold": 0.5
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/artwork.jpg", "threshold": 0.5}'
```

---

### 3. `POST /api/v1/detect/base64`
Detect from a Base64-encoded string or Data URI.

**Request Body**:
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "threshold": 0.5
}
```

---

### 4. `POST /api/v1/detect/batch`
Upload multiple files simultaneously for batch classification.

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/batch" \
  -F "files=@image1.png" \
  -F "files=@image2.jpg"
```

---

### 5. `GET /health` & `GET /api/v1/info`
Returns health check status and engine specifications.

---

## 🌐 Website Integration Guide

### 1. Vanilla JavaScript / Fetch API

Use this snippet in any web page to add an AI verification badge to user file uploads:

```html
<!-- HTML -->
<input type="file" id="imageInput" accept="image/*" />
<div id="detectionResult"></div>

<!-- JavaScript -->
<script>
const input = document.getElementById('imageInput');
const resultDiv = document.getElementById('detectionResult');

input.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  resultDiv.innerHTML = "⏳ Analyzing image with AI Image Detector...";

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8000/api/v1/detect?threshold=0.5', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Detection failed");
    }

    if (data.is_ai_generated) {
      resultDiv.innerHTML = `
        <div style="color: #ef4444; font-weight: bold;">
          🤖 Flagged as AI-Generated (${data.confidence_percentage} confidence)
        </div>
      `;
    } else {
      resultDiv.innerHTML = `
        <div style="color: #10b981; font-weight: bold;">
          👤 Verified Human-Created (${data.confidence_percentage} confidence)
        </div>
      `;
    }
  } catch (error) {
    resultDiv.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
  }
});
</script>
```

---

### 2. React / Next.js Integration

```tsx
import React, { useState } from 'react';

interface DetectionResponse {
  is_ai_generated: boolean;
  verdict: string;
  confidence_percentage: string;
  scores: { artificial: number; human: number };
  inference_time_ms: number;
}

export const ImageAIDetector: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/v1/detect?threshold=0.5', {
        method: 'POST',
        body: formData,
      });
      const data: DetectionResponse = await res.json();
      setResult(data);
    } catch (err) {
      console.error("AI detection error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-slate-900 text-white rounded-xl max-w-md mx-auto">
      <h3 className="text-xl font-bold mb-4">AI Image Authenticator</h3>
      <input type="file" accept="image/*" onChange={handleFileUpload} />

      {preview && (
        <img src={preview} alt="Upload preview" className="mt-4 rounded-lg max-h-48 w-full object-cover" />
      )}

      {loading && <p className="mt-4 text-indigo-400">Analyzing visual embeddings...</p>}

      {result && (
        <div className={`mt-4 p-4 rounded-lg ${result.is_ai_generated ? 'bg-red-950 border border-red-500' : 'bg-emerald-950 border border-emerald-500'}`}>
          <div className="font-bold text-lg">
            {result.is_ai_generated ? '🤖 AI-Generated Image' : '👤 Human-Created Image'}
          </div>
          <div className="text-sm opacity-90 mt-1">
            Confidence: {result.confidence_percentage} &middot; {result.inference_time_ms}ms
          </div>
        </div>
      )}
    </div>
  );
};
```

---

### 3. Vue.js 3 Integration

```html
<template>
  <div class="detector-card">
    <input type="file" @change="onFileChange" accept="image/*" />
    <div v-if="loading">Processing image...</div>
    <div v-else-if="result" :class="result.is_ai_generated ? 'ai-alert' : 'human-alert'">
      <h4>{{ result.verdict }} ({{ result.confidence_percentage }})</h4>
      <p>Artificial probability: {{ (result.scores.artificial * 100).toFixed(1) }}%</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const loading = ref(false);
const result = ref(null);

async function onFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  loading.value = true;
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('http://localhost:8000/api/v1/detect', {
      method: 'POST',
      body: formData,
    });
    result.value = await res.json();
  } catch (err) {
    alert("Detection failed: " + err.message);
  } finally {
    loading.value = false;
  }
}
</script>
```

---

### 4. Node.js / Express Backend Proxy

If you want your website frontend to talk to your own Express backend, and have Express proxy the check:

```javascript
const express = require('express');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios');

const upload = multer();
const app = express();

const DETECTOR_API_URL = process.env.DETECTOR_API_URL || 'http://localhost:8000';

app.post('/api/verify-image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${DETECTOR_API_URL}/api/v1/detect`, form, {
      headers: form.getHeaders(),
    });

    return res.json(response.data);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('Web server running on port 3000'));
```

---

### 5. Python Client

```python
import requests

API_URL = "http://localhost:8000/api/v1/detect"

# 1. File Upload
with open("sample.jpg", "rb") as f:
    response = requests.post(API_URL, files={"file": f}, params={"threshold": 0.5})
    data = response.json()
    print(f"Verdict: {data['verdict']}, Confidence: {data['confidence_percentage']}")

# 2. URL Detection
url_response = requests.post(
    "http://localhost:8000/api/v1/detect/url",
    json={"url": "https://example.com/art.png", "threshold": 0.5}
)
print("URL Result:", url_response.json())
```

---

## ⚙️ Extensive & Production Use Guide

When deploying this engine to handle thousands of requests per hour, follow these best practices:

### 1. Multi-Worker Concurrency (Gunicorn + Uvicorn)

For high-concurrency Linux/Unix production environments, run Uvicorn behind Gunicorn with multiple worker processes:

```bash
# Recommended worker count = (2 * CPU Cores) + 1
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

---

### 2. GPU Acceleration

If you have an NVIDIA GPU with CUDA:
1. Ensure NVIDIA CUDA Toolkit and cuDNN are installed.
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. The engine automatically detects `torch.cuda.is_available()` and offloads the Swin Transformer to GPU `cuda:0`.
4. **Latency comparison**:
   - **CPU**: ~220 ms per image
   - **NVIDIA RTX 4090 / A10G**: ~18–28 ms per image (10x faster)

---

### 3. High-Scale Batch & Queue Worker Architecture

For platforms where users upload galleries, avatars, or marketplace listings:
1. Don't block HTTP requests while evaluating hundreds of images.
2. Use an asynchronous message broker (Redis / RabbitMQ) with Celery workers:

```
[ User Upload ] ──> [ Web Backend ] ──> [ Redis Queue ] ──> [ Detector Worker (GPU) ]
                                                                      │
                                                             [ Store Verdict in DB ]
```

**Celery Task Example**:
```python
from celery import Celery
from engine import get_detector

app = Celery('detector_tasks', broker='redis://localhost:6379/0')
detector = get_detector()

@app.task
def process_user_upload(image_path: str, user_id: str):
    result = detector.predict(image_path)
    # Save result to database or flag moderation review
    if result["is_ai_generated"]:
        flag_for_moderation(user_id, image_path, result["confidence"])
    return result
```

---

### 4. Rate Limiting & API Security

To prevent DDoS or scraping abuse:
1. Configure an API Gateway (e.g. NGINX, Cloudflare, Traefik) or FastAPI rate limiter:
   ```bash
   pip install slowapi
   ```
2. Restrict CORS origins via environment variable in production:
   ```bash
   export CORS_ORIGINS="https://yourwebsite.com,https://app.yourwebsite.com"
   ```

---

### 5. Decision Threshold Calibration

The default threshold is `0.50`. You can tune it based on your platform's risk profile:
- **Strict Anti-AI Policy (e.g., Art competitions, copyright filters)**:
  - Set `threshold = 0.35` (flags potential AI art aggressively; higher false positive rate, zero tolerance for AI).
- **Balanced Social / Marketplace Platform**:
  - Set `threshold = 0.50` (standard balanced operating point).
- **High-Confidence Only (e.g., automated account banning)**:
  - Set `threshold = 0.75` (only flags images that exhibit overwhelming AI generative signatures to avoid penalizing legitimate human creators).

---

### 6. Model Capabilities & Limitations

- **Specialization**: The `umm-maybe/AI-image-detector` model was trained predominantly on artistic and synthetic visual styles (illustrations, concept art, 3D renders, painterly styles).
- **Ordinary Camera Photos**: Raw unedited phone camera pictures or non-artistic screenshots may sometimes yield lower confidence because they lack stylistic prompt signatures.
- **Ensemble Recommendation**: For enterprise-scale applications requiring detection across both photorealistic deepfakes and artistic generations, consider ensembling this Swin Transformer engine with an EXIF / frequency-domain analyzer (e.g. FFT noise analysis).

---

## 🧪 Running Automated Tests

Run the full automated test suite (covering unit tests, API tests, base64 conversion, threshold logic, and corruption handling):

```bash
# Run all test modules
python -m unittest discover -s tests -p "test_*.py"
```

All 13 tests execute in under 3 seconds!

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).  
The underlying model weights are distributed under the [Creative Commons Attribution 4.0 (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license by `umm-maybe`.
