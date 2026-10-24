"""
FastAPI Server for AI Image Detector & Refund Fraud Verification Engine.
Supports Sightengine GenAI Cloud API and Local ViT-Base Engine.
"""

import os
import sys
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# Configure UTF-8 stdout safely on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, Query, Header, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.detector import get_detector, AIImageDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing AI Image Detector engine...")
    detector = get_detector(force_reload=True)
    info = detector.get_info()
    print(f"[READY] Engine ready. Active provider: {info['active_provider']}, Sightengine configured: {info['sightengine_configured']}")
    yield
    print("[STOP] Shutting down AI Image Detector engine...")


app = FastAPI(
    title="AI Image Detector & Refund Fraud Verification Engine",
    description="""
Industry-grade REST API backend to detect AI-generated vs authentic images.
Integrates Sightengine's GenAI detection API with offline Local ViT-Base fallback,
Camera EXIF metadata auditing, and Return & Refund Fraud risk scoring.
    """,
    version="3.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Schemas
class URLDetectionRequest(BaseModel):
    url: str = Field(..., description="Direct HTTP or HTTPS URL to an image")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    provider: Optional[str] = Field("auto", description="'auto', 'sightengine', or 'local'")
    sightengine_user: Optional[str] = None
    sightengine_secret: Optional[str] = None
    include_forensics: Optional[bool] = Field(True, description="Compute 2D FFT, ELA, and sensor noise forensics")


class Base64DetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded string or Data URI")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    provider: Optional[str] = Field("auto")
    sightengine_user: Optional[str] = None
    sightengine_secret: Optional[str] = None
    include_forensics: Optional[bool] = Field(True, description="Compute 2D FFT, ELA, and sensor noise forensics")


class SightengineConfig(BaseModel):
    api_user: str
    api_secret: str


class ScoreBreakdown(BaseModel):
    artificial: float
    human: float


class ImageDimensions(BaseModel):
    width: int
    height: int


class EXIFMetadata(BaseModel):
    has_camera_exif: bool
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    date_time: Optional[str] = None
    software: Optional[str] = None
    all_tags_count: int


class RefundFraudAnalysis(BaseModel):
    risk_level: str
    recommended_action: str
    notes: str
    has_camera_exif: bool


class DetectionResponse(BaseModel):
    success: bool
    provider: str
    is_ai_generated: bool
    verdict: str
    confidence: float
    confidence_percentage: str
    scores: ScoreBreakdown
    threshold: float
    image_size: Optional[ImageDimensions] = None
    exif_metadata: Optional[EXIFMetadata] = None
    forensics: Optional[dict] = None
    refund_fraud_analysis: Optional[RefundFraudAnalysis] = None
    inference_time_ms: float
    model: str
    device: str
    error: Optional[str] = None


# Endpoints
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>AI Image Detector Engine is Running.</h2>")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "AI Image Detector & Refund Fraud Verification Engine",
    }


@app.get("/api/v1/info", tags=["Metadata"])
async def model_info():
    detector = get_detector()
    return detector.get_info()


@app.post("/api/v1/configure-sightengine", tags=["Configuration"])
async def configure_sightengine(config: SightengineConfig):
    """Dynamically set Sightengine credentials without restarting."""
    os.environ["SIGHTENGINE_API_USER"] = config.api_user.strip()
    os.environ["SIGHTENGINE_API_SECRET"] = config.api_secret.strip()
    detector = get_detector(
        force_reload=True,
        sightengine_user=config.api_user.strip(),
        sightengine_secret=config.api_secret.strip(),
    )
    return {
        "status": "success",
        "message": "Sightengine credentials saved successfully.",
        "sightengine_configured": True,
    }


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_uploaded_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    provider: str = Query("auto", description="'auto', 'sightengine', or 'local'"),
    include_forensics: bool = Query(True, description="Compute 2D FFT, ELA, and noise forensics"),
    x_sightengine_user: Optional[str] = Header(None, alias="X-Sightengine-User"),
    x_sightengine_secret: Optional[str] = Header(None, alias="X-Sightengine-Secret"),
):
    """Detect whether an uploaded image is AI-generated and assess refund fraud risk."""
    detector = get_detector()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = detector.predict(
            content,
            threshold=threshold,
            provider=provider,
            sightengine_user=x_sightengine_user,
            sightengine_secret=x_sightengine_secret,
            include_forensics=include_forensics,
        )
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to process image."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/forensics", tags=["Forensics"])
async def analyze_image_forensics(
    file: UploadFile = File(..., description="Image file to run digital forensics on"),
):
    """
    Run dedicated digital forensics analysis:
    - Metadata & Camera Hardware Audit (EXIF / C2PA / IPTC)
    - 2D Fast Fourier Transform (FFT) Neural Grid Artifact Detection
    - Error Level Analysis (ELA) Compression Tampering Heatmap
    - Noise Residuals & Diffusion Smoothing Analysis (Laplacian)
    """
    from PIL import Image
    import io
    from engine.forensics import DigitalForensicsAnalyzer

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        img = Image.open(io.BytesIO(content))
        analyzer = DigitalForensicsAnalyzer()
        report = analyzer.run_full_forensics(img, raw_bytes=content)
        report["filename"] = file.filename
        report["image_size"] = {"width": img.width, "height": img.height}
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")


@app.post("/api/v1/detect/url", response_model=DetectionResponse, tags=["Detection"])
async def detect_image_url(request: URLDetectionRequest):
    detector = get_detector()
    try:
        result = detector.predict(
            request.url,
            threshold=request.threshold,
            provider=request.provider,
            sightengine_user=request.sightengine_user,
            sightengine_secret=request.sightengine_secret,
            include_forensics=request.include_forensics if request.include_forensics is not None else True,
        )
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to analyze URL."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/detect/base64", response_model=DetectionResponse, tags=["Detection"])
async def detect_image_base64(request: Base64DetectionRequest):
    detector = get_detector()
    try:
        result = detector.predict(
            request.image_base64,
            threshold=request.threshold,
            provider=request.provider,
            sightengine_user=request.sightengine_user,
            sightengine_secret=request.sightengine_secret,
            include_forensics=request.include_forensics if request.include_forensics is not None else True,
        )
        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Failed to decode base64."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/detect/batch", tags=["Detection"])
async def detect_batch_images(
    files: List[UploadFile] = File(..., description="Multiple image files to classify"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    provider: str = Query("auto", description="'auto', 'sightengine', or 'local'"),
    x_sightengine_user: Optional[str] = Header(None, alias="X-Sightengine-User"),
    x_sightengine_secret: Optional[str] = Header(None, alias="X-Sightengine-Secret"),
):
    """Batch detection: classify multiple uploaded images simultaneously."""
    detector = get_detector()
    results = []

    for f in files:
        try:
            content = await f.read()
            res = detector.predict(
                content,
                threshold=threshold,
                provider=provider,
                sightengine_user=x_sightengine_user,
                sightengine_secret=x_sightengine_secret,
            )
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
