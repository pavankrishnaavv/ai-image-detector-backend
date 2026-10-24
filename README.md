# 🛡️ AI Image Detector & Refund Fraud Verification Engine
### Powered by Sightengine GenAI API + Local ViT + Digital Forensics Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Sightengine](https://img.shields.io/badge/API-Sightengine%20GenAI-blueviolet.svg)](https://sightengine.com)
[![Digital Forensics](https://img.shields.io/badge/Forensics-2D%20FFT%20%7C%20ELA%20%7C%20Noise-orange.svg)](https://github.com/pavankrishnaavv/ai-image-detector-backend)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready backend engine and REST API designed to detect **AI-generated images** (Midjourney, DALL-E 3, FLUX, Stable Diffusion) and protect platforms against **return & refund fraud** (fake damaged products, AI-inpainted cracks, synthetic claim receipts).

Features a **Hybrid Architecture**:
- ⚡ **Sightengine GenAI Cloud API**: Industry-leading accuracy (90-99%+ detection on modern DALL-E 3, FLUX, Midjourney v6). Includes **2,000 free operations/month**.
- 🔬 **Digital Forensics Suite (Frequency & Spatial Residuals)**:
  - **2D Fast Fourier Transform (FFT)**: Identifies high-frequency periodic grid spikes and checkerboard artifacts left by transposed convolutions & neural upsamplers.
  - **Error Level Analysis (ELA)**: Recompression delta analysis detecting localized inpainting, Photoshop Generative Fill, and synthetic tampering.
  - **Laplacian Sensor Noise Consistency**: Distinguishes true physical sensor photon noise (PRNU) from unnatural diffusion denoising smoothing.
  - **C2PA & Provenance Audit**: Inspects Content Credentials (`c2pa`), Adobe JUMBF manifests, Google SynthID markers, and IPTC `trainedAlgorithmicMedia`.
- 💻 **Offline Local Vision Transformer**: Bundled local Google ViT model (`dima806/ai_vs_real_image_detection`) for zero-cost, offline fallback.
- 📷 **Camera Sensor EXIF Audit**: Verifies real smartphone lens metadata (`Apple iPhone`, `Samsung Galaxy`, `FocalLength`, `ShutterSpeed`) to catch downloaded or synthetic files.
- 🛡️ **Automated Refund Triage**: Classifies claims into `HIGH_RISK`, `MEDIUM_RISK`, or `LOW_RISK` with policy action recommendations.

---

## 📑 Table of Contents
1. [Sightengine Setup (Free Tier)](#-sightengine-setup-free-tier)
2. [Quickstart & Installation](#-quickstart--installation)
3. [Interactive Web Dashboard](#-interactive-web-dashboard)
4. [CLI Tool Usage](#-cli-tool-usage)
5. [REST API Documentation](#-rest-api-documentation)
6. [Website Integration Guide](#-website-integration-guide)
   - [Vanilla JavaScript (Return Portal)](#1-vanilla-javascript--return-portal)
   - [React / Next.js Component](#2-react--nextjs-integration)
   - [Node.js / Express Proxy](#3-nodejs--express-backend-proxy)
   - [Python Client](#4-python-client)
7. [Extensive & Production Use Guide](#-extensive--production-use-guide)
8. [Running Automated Tests](#-running-automated-tests)
9. [License](#-license)

---

## 🔑 Sightengine Setup (Free Tier)

Sightengine provides **2,000 free operations every month** with no credit card required.

1. **Sign Up**: Go to [https://sightengine.com/](https://sightengine.com/) and register a free account.
2. **Get Credentials**: Copy your **`api_user`** and **`api_secret`** from the dashboard.
3. **Configure the Backend**:
   - **Method A (via `.env`)**:
     Open `.env` in the repository root and add:
     ```env
     SIGHTENGINE_API_USER=your_api_user_here
     SIGHTENGINE_API_SECRET=your_api_secret_here
     ```
   - **Method B (via Web Dashboard)**:
     Open `http://localhost:8000/`, click **⚙️ Sightengine Key** in the top-right header, paste your credentials, and click **Save & Activate**.
   - **Method C (per API Request)**:
     Pass headers `X-Sightengine-User` and `X-Sightengine-Secret` with any API call.

*(If Sightengine credentials are not provided, the engine automatically uses the bundled local ViT model as an offline fallback!)*

---

## 🏗️ Project Structure

```text
ai-image-detector-backend/
├── engine/
│   ├── __init__.py          # Package initialization
│   ├── detector.py          # Unified Hybrid Detection Engine + EXIF + Refund Triage
│   └── sightengine_client.py# Sightengine GenAI API connector
├── model_weights/           # Bundled offline ViT model weights & preprocessor
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors    # Tracked with Git LFS
├── static/
│   └── index.html           # Interactive dashboard with Sightengine settings modal
├── tests/
│   ├── __init__.py
│   ├── test_engine.py       # Engine unit tests
│   ├── test_api.py          # FastAPI endpoint integration tests
│   ├── generate_test_images.py
│   └── sample_images/       # Test fixtures
├── cli.py                   # Terminal testing tool
├── main.py                  # FastAPI REST API server
├── save_model.py            # Model export utility
├── requirements.txt         # Dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose setup
├── .env.example             # Environment template
└── README.md                # Documentation
```

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/ai-image-detector-backend.git
cd ai-image-detector-backend

# 2. Setup virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux / macOS

# 3. Install requirements
pip install -r requirements.txt

# 4. Configure .env (Optional: add Sightengine keys)
cp .env.example .env

# 5. Start the backend
python main.py
```

Open:
- **Interactive Web UI**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖥️ Interactive Web Dashboard

Open `http://localhost:8000/`:
- **Drag-and-Drop Image Uploader**: Drop any damaged item or customer claim photo.
- **Provider Status Indicator**: Displays whether Sightengine API or Local ViT is currently active.
- **Sightengine Key Modal**: Configure credentials directly in the UI without restarting.
- **Refund & Return Risk Box**: Instant triage (`HIGH RISK`, `MEDIUM RISK`, `LOW RISK`) with policy recommendations.
- **Camera EXIF Inspection**: Verifies if genuine camera sensor tags are present.

---

## 💻 CLI Tool Usage

```bash
# Analyze using Auto Mode (uses Sightengine if keys configured, else local ViT)
python cli.py path/to/damage_photo.jpg

# Force Sightengine with credentials
python cli.py path/to/damage_photo.jpg --provider sightengine --sightengine-user "USER_ID" --sightengine-secret "SECRET"

# Force Local ViT
python cli.py path/to/damage_photo.jpg --provider local

# Output raw JSON
python cli.py path/to/damage_photo.jpg --json
```

**CLI Output Example**:
```text
==============================================================
Provider:             Sightengine GenAI API
Verdict:              [AI-GENERATED] AI-Generated
Confidence:           90.00%
AI / Fake Score:      90.00%
Real / Authentic:     10.00%
Decision Threshold:   0.5
Image Resolution:     1024x538 px
Inference Latency:    1879.31 ms
Model:                sightengine/genai
Execution Target:     CLOUD_API
--------------------------------------------------------------
DIGITAL FORENSICS & NEURAL RESIDUAL AUDIT:
  Forensic Verdict:   HIGH_SYNTHETIC_PROBABILITY
  Forensic AI Score:  68.0%
  2D FFT Grid Spikes: 80 peaks (Checkerboard: True)
  Noise Inconsistency:1.117 (Diffusion Smoothing: True)
  ELA Compression:    Mean 0.811, Max 18.0 (Tampering: False)
  Camera EXIF Audit:  No physical camera hardware EXIF tags found. Metadata is stripped or missing, typical of AI generation or web downloads.
  Key Forensic Flags:
    * Metadata: Camera sensor hardware EXIF is missing/stripped.
    * FFT 2D Spectrum: 80 periodic grid spikes detected (neural upsampling artifact).
    * Noise Residuals: High noise inconsistency index (1.117), characteristic of diffusion denoising.
--------------------------------------------------------------
RETURN / REFUND CLAIM ANALYSIS:
  Risk Level:         HIGH_RISK
  Recommended Action: FLAGGED_FOR_MANUAL_REVIEW
  Camera EXIF Found:  False
  Policy Notes:       High confidence synthetic / AI-generated image. Periodic 2D FFT neural upsampling artifacts detected.
==============================================================
```

---

## 🔬 How Digital Forensics & Neural Residuals Work

Unlike simple classification models that only evaluate artistic style, our **Digital Forensics Suite** examines the low-level physical and mathematical properties of digital images:

1. **2D Fast Fourier Transform (FFT) Neural Grid Spikes**:
   - Neural generators (StyleGAN, Stable Diffusion, DALL-E) construct images using convolutional layers, transposed convolutions, or patch upsamplers.
   - These discrete upsampling grids leave behind **periodic spatial harmonics** (checkerboard artifacts).
   - Natural photographs exhibit a smooth power-law radial frequency decay ($1/f^\alpha$). Synthetic images exhibit isolated harmonic peaks in the high-frequency spectrum.
2. **Error Level Analysis (ELA)**:
   - When an image is recompressed at a fixed JPEG quantization table (quality 90%), differences between the original and recompressed pixels reveal compression history.
   - In genuine photos, surfaces of similar texture show uniform error. In AI-inpainted images (e.g., synthetic cracks or broken keys inserted into a photo), the modified sections display distinct quantization deltas.
3. **Sensor Noise Residuals & Laplacian Variance**:
   - Authentic camera sensors have physical Photo-Response Non-Uniformity (PRNU) and photon shot noise evenly distributed across the entire sensor frame.
   - Diffusion models generate images by iteratively removing Gaussian noise, leaving unnatural spatial smoothing on flat surfaces juxtaposed against high pseudo-noise on textured surfaces.
4. **Metadata & Provenance Verification**:
   - Audits camera hardware EXIF trees (`Make`, `Model`, `Lens`, `FocalLength`, `ShutterSpeed`, `ISO`).
   - Scans raw JPEG/PNG byte streams for C2PA JUMBF boxes, Adobe Content Credentials, Google SynthID watermarks, and IPTC `trainedAlgorithmicMedia`.

---

## 📡 REST API Documentation

### `POST /api/v1/detect`
Upload an image via `multipart/form-data`.

**Parameters**:
- `file` (form-data, required): Binary image file (`.png`, `.jpg`, `.webp`).
- `threshold` (query param, optional, default `0.5`): AI decision threshold.
- `provider` (query param, optional, default `"auto"`): `"auto"`, `"sightengine"`, or `"local"`.

**Headers (Optional)**:
- `X-Sightengine-User`: Sightengine API user ID.
- `X-Sightengine-Secret`: Sightengine API secret.

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect?threshold=0.5" \
  -F "file=@keyboard.jpg"
```

**Response (`200 OK`)**:
```json
{
  "success": true,
  "provider": "Sightengine GenAI API",
  "is_ai_generated": true,
  "verdict": "AI-Generated",
  "confidence": 0.99,
  "confidence_percentage": "99.00%",
  "scores": {
    "artificial": 0.99,
    "human": 0.01
  },
  "threshold": 0.5,
  "image_size": {
    "width": 1024,
    "height": 538
  },
  "exif_metadata": {
    "has_camera_exif": false,
    "camera_make": null,
    "camera_model": null,
    "date_time": null,
    "software": null,
    "all_tags_count": 0
  },
  "refund_fraud_analysis": {
    "risk_level": "HIGH_RISK",
    "recommended_action": "FLAGGED_FOR_MANUAL_REVIEW",
    "notes": "High confidence synthetic / AI-generated image. Reject or flag claim for manual inspection.",
    "has_camera_exif": false
  },
  "inference_time_ms": 182.4,
  "model": "sightengine/genai",
  "device": "cloud_api",
  "error": null
}
```

---

## 🌐 Website Integration Guide

### 1. Vanilla JavaScript (Return Portal)

```javascript
async function verifyReturnPhoto(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('http://localhost:8000/api/v1/detect', {
    method: 'POST',
    body: formData,
  });
  const data = await res.json();

  if (data.refund_fraud_analysis.risk_level === 'HIGH_RISK') {
    alert("⚠️ Suspected synthetic image detected. Your refund claim will be routed to a specialist.");
  } else {
    alert("✅ Photo verified. Proceeding with instant refund.");
  }
}
```

---

### 2. React / Next.js Component

```tsx
import React, { useState } from 'react';

export const RefundClaimUploader: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('http://localhost:8000/api/v1/detect', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="p-6 bg-slate-900 text-white rounded-xl max-w-lg">
      <h3 className="text-xl font-bold mb-2">Upload Damage Proof</h3>
      <input type="file" accept="image/*" onChange={handleUpload} />

      {loading && <p className="mt-3 text-indigo-400">Verifying authenticity via Sightengine...</p>}

      {result && (
        <div className={`mt-4 p-4 rounded-lg ${result.is_ai_generated ? 'bg-red-950 border border-red-500' : 'bg-emerald-950 border border-emerald-500'}`}>
          <div className="font-bold">
            {result.is_ai_generated ? '⚠️ AI-Generated Damage Detected' : '✅ Verified Authentic Photo'}
          </div>
          <div className="text-sm mt-1">
            Provider: {result.provider} &middot; Risk: {result.refund_fraud_analysis.risk_level}
          </div>
        </div>
      )}
    </div>
  );
};
```

---

### 3. Node.js / Express Backend Proxy

```javascript
const express = require('express');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios');

const upload = multer();
const app = express();

app.post('/api/claims/verify', upload.single('photo'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const check = await axios.post('http://localhost:8000/api/v1/detect', form, {
      headers: form.getHeaders(),
    });

    const analysis = check.data.refund_fraud_analysis;

    if (analysis.risk_level === 'HIGH_RISK') {
      return res.json({ status: 'HOLD_REFUND', reason: analysis.notes });
    }

    return res.json({ status: 'AUTO_REFUND_APPROVED' });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

app.listen(3000);
```

---

## 🧪 Running Automated Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 13 tests execute and pass in ~2 seconds.

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).  
Sightengine is a registered trademark of Sightengine SAS.
