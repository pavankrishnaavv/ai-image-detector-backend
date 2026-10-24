"""
Unified AI Image Detector Engine with Sightengine & Local ViT Support.
Specialized in detecting AI-generated/synthetic media and return/refund fraud.
"""

import os
import io
import time
import base64
import logging
from typing import List, Dict, Any, Union, Optional
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
import torch
from transformers import pipeline

from .sightengine_client import SightengineClient

logger = logging.getLogger("ai_detector.engine")

DEFAULT_MODEL_NAME = "dima806/ai_vs_real_image_detection"
LOCAL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_weights"
)


class AIImageDetector:
    """
    Hybrid inference engine supporting both Sightengine Cloud API (industry standard)
    and offline Local Vision Transformer (ViT-Base).
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        sightengine_user: Optional[str] = None,
        sightengine_secret: Optional[str] = None,
        preferred_provider: str = "auto",  # 'auto', 'sightengine', or 'local'
    ):
        self.threshold = threshold
        self.preferred_provider = preferred_provider

        # Setup Sightengine Client
        self.sightengine = SightengineClient(
            api_user=sightengine_user,
            api_secret=sightengine_secret,
        )

        # Device selection for local model
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Local model path
        if model_name_or_path:
            self.model_path = model_name_or_path
        elif (
            os.path.exists(LOCAL_WEIGHTS_PATH)
            and os.path.exists(os.path.join(LOCAL_WEIGHTS_PATH, "config.json"))
            and os.path.exists(os.path.join(LOCAL_WEIGHTS_PATH, "model.safetensors"))
        ):
            self.model_path = LOCAL_WEIGHTS_PATH
        else:
            self.model_path = DEFAULT_MODEL_NAME

        self._local_pipeline = None

    def _ensure_local_pipeline(self):
        """Lazy load local ViT model if needed."""
        if self._local_pipeline is None:
            logger.info(f"Loading local ViT pipeline from '{self.model_path}' on {self.device}...")
            device_index = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else self.device)
            self._local_pipeline = pipeline(
                "image-classification",
                model=self.model_path,
                device=device_index,
            )

    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """Extracts camera EXIF metadata to audit customer photo claims."""
        metadata = {
            "has_camera_exif": False,
            "camera_make": None,
            "camera_model": None,
            "date_time": None,
            "software": None,
            "all_tags_count": 0,
        }
        try:
            exif = img.getexif()
            if exif:
                metadata["all_tags_count"] = len(exif)
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make":
                        metadata["camera_make"] = str(value).strip()
                    elif tag_name == "Model":
                        metadata["camera_model"] = str(value).strip()
                    elif tag_name in ("DateTime", "DateTimeOriginal"):
                        metadata["date_time"] = str(value).strip()
                    elif tag_name == "Software":
                        metadata["software"] = str(value).strip()

                if metadata["camera_make"] or metadata["camera_model"]:
                    metadata["has_camera_exif"] = True
        except Exception as e:
            logger.warning(f"Failed to parse EXIF: {e}")

        return metadata

    def _normalize_image(self, image_input: Union[Image.Image, bytes, str]):
        """Normalizes input into PIL.Image, raw bytes, and EXIF metadata."""
        raw_bytes = None
        url = None

        if isinstance(image_input, Image.Image):
            img = image_input
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            raw_bytes = buf.getvalue()
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, str):
            if image_input.startswith("data:image/") or ";base64," in image_input:
                b64_clean = image_input.split(";base64,")[-1]
                raw_bytes = base64.b64decode(b64_clean)
                img = Image.open(io.BytesIO(raw_bytes))
            elif image_input.startswith("http://") or image_input.startswith("https://"):
                url = image_input
                import urllib.request
                req = urllib.request.Request(
                    image_input,
                    headers={"User-Agent": "AI-Image-Detector-Backend/3.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    raw_bytes = response.read()
                img = Image.open(io.BytesIO(raw_bytes))
            elif os.path.isfile(image_input):
                with open(image_input, "rb") as f:
                    raw_bytes = f.read()
                img = Image.open(io.BytesIO(raw_bytes))
            else:
                try:
                    raw_bytes = base64.b64decode(image_input)
                    img = Image.open(io.BytesIO(raw_bytes))
                except Exception:
                    raise ValueError("Unsupported image input format.")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        exif_info = self._extract_exif(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img, raw_bytes, url, exif_info

    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        threshold: Optional[float] = None,
        provider: Optional[str] = None,
        sightengine_user: Optional[str] = None,
        sightengine_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classifies an image using Sightengine API (if configured) or local ViT.
        """
        eff_threshold = threshold if threshold is not None else self.threshold
        start_time = time.perf_counter()

        try:
            image, raw_bytes, remote_url, exif_info = self._normalize_image(image_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process image: {str(e)}",
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Determine provider to use
        req_provider = provider or self.preferred_provider
        
        # Override sightengine credentials if provided in request
        client = self.sightengine
        if sightengine_user and sightengine_secret:
            client = SightengineClient(api_user=sightengine_user, api_secret=sightengine_secret)

        use_sightengine = False
        if req_provider == "sightengine":
            use_sightengine = True
        elif req_provider == "auto" and client.is_configured:
            use_sightengine = True

        ai_prob = 0.0
        active_provider_name = ""

        if use_sightengine:
            active_provider_name = "Sightengine GenAI API"
            if remote_url:
                res = client.detect_url(remote_url)
            else:
                res = client.detect_bytes(raw_bytes)

            if not res["success"]:
                # If Sightengine failed (e.g. invalid keys), log and fallback to local if in auto mode
                logger.warning(f"Sightengine call failed ({res.get('error')}). Falling back to local model.")
                if req_provider == "auto":
                    self._ensure_local_pipeline()
                    local_res = self._local_pipeline(image)
                    scores = {item["label"].upper(): float(item["score"]) for item in local_res}
                    fake = scores.get("FAKE", scores.get("ARTIFICIAL", 0.0))
                    ai_prob = round(fake, 4)
                    active_provider_name = "Local ViT Engine (Fallback from Sightengine)"
                else:
                    return {
                        "success": False,
                        "error": f"Sightengine error: {res.get('error')}",
                        "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    }
            else:
                ai_prob = res["ai_score"]
        else:
            active_provider_name = "Local ViT Engine (dima806/ai_vs_real_image_detection)"
            self._ensure_local_pipeline()
            raw_results = self._local_pipeline(image)
            scores = {item["label"].upper(): float(item["score"]) for item in raw_results}
            fake_score = scores.get("FAKE", scores.get("ARTIFICIAL", 0.0))
            real_score = scores.get("REAL", scores.get("HUMAN", 0.0))
            total = fake_score + real_score
            ai_prob = round(fake_score / total, 4) if total > 0 else round(fake_score, 4)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        real_prob = round(1.0 - ai_prob, 4)
        is_ai = ai_prob >= eff_threshold
        verdict = "AI-Generated" if is_ai else "Human-Created (Real)"
        confidence = ai_prob if is_ai else real_prob

        # Refund & Return Fraud Risk Scoring
        if is_ai and ai_prob >= 0.70:
            refund_risk = "HIGH_RISK"
            refund_action = "FLAGGED_FOR_MANUAL_REVIEW"
            refund_notes = "High confidence synthetic / AI-generated image. Reject or flag claim for manual inspection."
        elif is_ai or (ai_prob >= 0.40 and not exif_info["has_camera_exif"]):
            refund_risk = "MEDIUM_RISK"
            refund_action = "REQUEST_ADDITIONAL_PROOF"
            refund_notes = "Borderline synthetic signature or missing camera EXIF. Request receipt or packaging video."
        else:
            refund_risk = "LOW_RISK"
            refund_action = "APPROVED_FOR_PROCESSING"
            refund_notes = "Genuine customer camera capture confirmed."

        return {
            "success": True,
            "provider": active_provider_name,
            "is_ai_generated": is_ai,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "scores": {
                "artificial": ai_prob,
                "human": real_prob,
            },
            "threshold": eff_threshold,
            "image_size": {
                "width": image.width,
                "height": image.height,
            },
            "exif_metadata": exif_info,
            "refund_fraud_analysis": {
                "risk_level": refund_risk,
                "recommended_action": refund_action,
                "notes": refund_notes,
                "has_camera_exif": exif_info["has_camera_exif"],
            },
            "inference_time_ms": inference_time_ms,
            "model": "sightengine/genai" if "Sightengine" in active_provider_name else "dima806/ai_vs_real_image_detection",
            "device": "cloud_api" if "Sightengine" in active_provider_name else self.device,
        }

    def predict_batch(self, images: List[Union[Image.Image, bytes, str]], **kwargs) -> List[Dict[str, Any]]:
        return [self.predict(img, **kwargs) for img in images]

    def get_info(self) -> Dict[str, Any]:
        sightengine_configured = self.sightengine.is_configured
        return {
            "model_name": "dima806/ai_vs_real_image_detection",
            "labels": ["REAL", "FAKE"],
            "model_source": "local_weights" if self.model_path == LOCAL_WEIGHTS_PATH else "huggingface",
            "active_provider": "Sightengine GenAI API" if sightengine_configured else "Local ViT Engine",
            "sightengine_configured": sightengine_configured,
            "local_model": "dima806/ai_vs_real_image_detection",
            "device": self.device,
            "default_threshold": self.threshold,
            "features": [
                "Sightengine GenAI API Support (2,000 free ops/mo)",
                "Local ViT-Base Offline Fallback",
                "Camera Sensor EXIF Extraction",
                "E-Commerce Refund Fraud Triage",
            ],
        }


_GLOBAL_DETECTOR: Optional[AIImageDetector] = None


def get_detector(force_reload: bool = False, **kwargs) -> AIImageDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None or force_reload:
        _GLOBAL_DETECTOR = AIImageDetector(**kwargs)
    return _GLOBAL_DETECTOR
