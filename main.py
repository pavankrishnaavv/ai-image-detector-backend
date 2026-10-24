"""
FastAPI Server for AI Image Detector Backend.
Provides high-performance REST APIs for detecting AI-generated images using
the 'umm-maybe/AI-image-detector' model.
"""

import os
import sys
import time
from typing import List, Optional
from contextlib import asynccontextmanager

# Configure UTF-8 stdout safely on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl

# Ensure parent directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.detector import get_detector, AIImageDetector

# Model warm-up on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing AI Image Detector engine...")
    detector = get_detector()
    info = detector.get_info()
    print(f"[READY] Detector engine ready. Source: {info['model_source']}, Device: {info['device']}")
    yield
    print("[STOP] Shutting down AI Image Detector engine...")


app = FastAPI(
    title="AI Image Detector Backend Engine",
    description="""
High-throughput REST API backend engine to detect AI-generated vs human-created images 
using the Hugging Face `umm-maybe/AI-image-detector` Swin Transformer architecture.

### Supported Input Modes:
- 📤 **File Upload (`multipart/form-data`)**: Upload raw image files (PNG, JPG, WebP, etc.)
- 🔗 **Image URL (`application/json`)**: Analyze any publicly accessible image link
- 🔤 **Base64 String (`application/json`)**: Send data URI or raw base64 encoded strings
- 📦 **Batch Analysis**: Analyze multiple images in a single call for high-scale workflows
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for universal integration across frontend websites
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for the test UI
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Schemas
class URLDetectionRequest(BaseModel):
    url: str = Field(..., description="Direct HTTP or HTTPS URL to an image")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for AI generation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://images.unsplash.com/photo-1546768292-fb12f6c92568",
                "threshold": 0.5
            }
        }
    }


class Base64DetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded string or Data URI (e.g. data:image/jpeg;base64,...)")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for AI generation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "threshold": 0.5
            }
        }
    }


class ScoreBreakdown(BaseModel):
    artificial: float
    human: float


class ImageDimensions(BaseModel):
    width: int
    height: int


class DetectionResponse(BaseModel):
    success: bool
    is_ai_generated: bool
    verdict: str
    confidence: float
    confidence_percentage: str
    scores: ScoreBreakdown
    threshold: float
    image_size: Optional[ImageDimensions] = None
    inference_time_ms: float
    model: str
    device: str
    error: Optional[str] = None


# Endpoints
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def index():
    """Serves the interactive web test interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>AI Image Detector Engine is Running. Visit <a href='/docs'>/docs</a> for API.</h2>")


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health status check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "AI Image Detector Engine",
    }


@app.get("/api/v1/info", tags=["Metadata"])
async def model_info():
    """Returns engine specifications, model architecture, and operational metadata."""
    detector = get_detector()
    return detector.get_info()


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_uploaded_image(
    file: UploadFile = File(..., description="Image file to analyze (PNG, JPG, WebP, BMP, etc.)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Sensitivity threshold for AI detection"),
):
    """
    Detect whether an uploaded image file is AI-generated.
    """
    detector = get_detector()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        result = detector.predict(content, threshold=threshold)
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to process image."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/detect/url", response_model=DetectionResponse, tags=["Detection"])
async def detect_image_url(request: URLDetectionRequest):
    """
    Detect whether an image from a public URL is AI-generated.
    """
    detector = get_detector()
    try:
        result = detector.predict(request.url, threshold=request.threshold)
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to fetch or process image URL."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/detect/base64", response_model=DetectionResponse, tags=["Detection"])
async def detect_image_base64(request: Base64DetectionRequest):
    """
    Detect whether a base64-encoded image is AI-generated.
    """
    detector = get_detector()
    try:
        result = detector.predict(request.image_base64, threshold=request.threshold)
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to decode or process base64 image."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/detect/batch", tags=["Detection"])
async def detect_batch_images(
    files: List[UploadFile] = File(..., description="Multiple image files to classify"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold"),
):
    """
    Batch detection: classify multiple uploaded images simultaneously.
    """
    detector = get_detector()
    results = []

    for f in files:
        try:
            content = await f.read()
            res = detector.predict(content, threshold=threshold)
            res["filename"] = f.filename
            results.append(res)
        except Exception as e:
            results.append({
                "filename": f.filename,
                "success": False,
                "error": str(e),
            })

    return {
        "total_images": len(files),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[INFO] Starting server on http://{host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=False)
