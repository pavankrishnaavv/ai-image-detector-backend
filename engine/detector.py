"""
Core AI Image Detector Engine.
Powered by 'dima806/ai_vs_real_image_detection' Vision Transformer (ViT) architecture.
Specialized in distinguishing AI-generated/synthetic media from authentic photographs,
including e-commerce damage claims and return/refund verification.
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

logger = logging.getLogger("ai_detector.engine")

DEFAULT_MODEL_NAME = "dima806/ai_vs_real_image_detection"
LOCAL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_weights"
)


class AIImageDetector:
    """
    Inference engine for AI Image Detection with Return/Refund Fraud Verification.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the detector engine.
        :param model_name_or_path: Path to local directory or Hugging Face model id.
        :param device: 'cuda', 'mps', 'cpu' or None (auto-detect).
        :param threshold: Classification threshold for AI/Fake label (default 0.5).
        """
        self.threshold = threshold

        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Determine model path: prefer local_weights if populated, else HF hub
        if model_name_or_path:
            self.model_path = model_name_or_path
        elif (
            os.path.exists(LOCAL_WEIGHTS_PATH)
            and os.path.exists(os.path.join(LOCAL_WEIGHTS_PATH, "config.json"))
            and os.path.exists(os.path.join(LOCAL_WEIGHTS_PATH, "model.safetensors"))
        ):
            self.model_path = LOCAL_WEIGHTS_PATH
            logger.info(f"Using local model weights from: {self.model_path}")
        else:
            self.model_path = DEFAULT_MODEL_NAME
            logger.info(f"Using Hugging Face model: {self.model_path}")

        self._pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Loads the image classification pipeline."""
        start_time = time.time()
        logger.info(f"Loading pipeline from '{self.model_path}' on device '{self.device}'...")

        device_index = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else self.device)
        self._pipeline = pipeline(
            "image-classification",
            model=self.model_path,
            device=device_index,
        )
        load_time = time.time() - start_time
        logger.info(f"Model successfully loaded in {load_time:.2f} seconds.")

    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """
        Extracts camera EXIF metadata to cross-verify real vs synthetic claims.
        Authentic customer smartphone photos typically contain rich EXIF tags.
        """
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
        """
        Converts any supported input format into a RGB PIL.Image and extracts raw metadata.
        """
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            # Check for base64
            if image_input.startswith("data:image/") or ";base64," in image_input:
                b64_clean = image_input.split(";base64,")[-1]
                img_data = base64.b64decode(b64_clean)
                img = Image.open(io.BytesIO(img_data))
            elif image_input.startswith("http://") or image_input.startswith("https://"):
                import urllib.request
                req = urllib.request.Request(
                    image_input,
                    headers={"User-Agent": "AI-Image-Detector-Backend/2.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    img_data = response.read()
                img = Image.open(io.BytesIO(img_data))
            elif os.path.isfile(image_input):
                img = Image.open(image_input)
            else:
                try:
                    img_data = base64.b64decode(image_input)
                    img = Image.open(io.BytesIO(img_data))
                except Exception:
                    raise ValueError(
                        "Unsupported image string input. Must be a valid file path, URL, or base64 data URI."
                    )
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        exif_info = self._extract_exif(img)

        # Ensure RGB mode (handles RGBA, Palette, Grayscale, CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, exif_info

    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform detection on a single image.
        :param image_input: PIL Image, bytes, base64 string, URL, or filepath.
        :param threshold: Custom decision threshold (defaults to self.threshold).
        :return: Structured prediction dictionary with refund fraud assessment.
        """
        eff_threshold = threshold if threshold is not None else self.threshold
        start_time = time.perf_counter()

        try:
            image, exif_info = self._normalize_image(image_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process image: {str(e)}",
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Run model inference
        raw_results = self._pipeline(image)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Parse classification scores (normalize keys across ViT / Swin models)
        scores = {}
        for item in raw_results:
            scores[item["label"].upper()] = float(item["score"])

        # Map labels (FAKE / ARTIFICIAL vs REAL / HUMAN)
        fake_score = scores.get("FAKE", scores.get("ARTIFICIAL", 0.0))
        real_score = scores.get("REAL", scores.get("HUMAN", 0.0))

        total = fake_score + real_score
        if total > 0:
            ai_prob = round(fake_score / total, 4)
            real_prob = round(real_score / total, 4)
        else:
            ai_prob = round(fake_score, 4)
            real_prob = round(real_score, 4)

        is_ai = ai_prob >= eff_threshold
        verdict = "AI-Generated" if is_ai else "Human-Created (Real)"
        confidence = ai_prob if is_ai else real_prob

        # Refund & Return Fraud Risk Scoring
        if is_ai and ai_prob >= 0.75:
            refund_risk = "HIGH_RISK"
            refund_action = "FLAGGED_FOR_MANUAL_REVIEW"
            refund_notes = "High confidence AI-generated damage or synthetic media detected. Do not auto-refund."
        elif is_ai or (ai_prob >= 0.40 and not exif_info["has_camera_exif"]):
            refund_risk = "MEDIUM_RISK"
            refund_action = "REQUEST_ADDITIONAL_PROOF"
            refund_notes = "Moderate synthetic score or missing camera sensor EXIF. Request receipt / live video."
        else:
            refund_risk = "LOW_RISK"
            refund_action = "APPROVED_FOR_PROCESSING"
            refund_notes = "Authentic photo signatures consistent with real customer camera capture."

        return {
            "success": True,
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
            "model": "dima806/ai_vs_real_image_detection",
            "device": self.device,
        }

    def predict_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform detection on a batch of images.
        """
        results = []
        for img in images:
            results.append(self.predict(img, threshold=threshold))
        return results

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the active engine and model."""
        return {
            "model_name": "dima806/ai_vs_real_image_detection",
            "architecture": "ViTForImageClassification (Google ViT-Base)",
            "labels": ["REAL", "FAKE"],
            "device": self.device,
            "model_source": "local_weights" if self.model_path == LOCAL_WEIGHTS_PATH else "huggingface",
            "default_threshold": self.threshold,
            "features": ["AI Image Detection", "Camera EXIF Audit", "Refund Fraud Triage"],
        }


# Global singleton instance
_GLOBAL_DETECTOR: Optional[AIImageDetector] = None


def get_detector(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
    force_reload: bool = False,
) -> AIImageDetector:
    """
    Get or create the global singleton detector instance.
    """
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None or force_reload:
        _GLOBAL_DETECTOR = AIImageDetector(
            model_name_or_path=model_name_or_path,
            device=device,
            threshold=threshold,
        )
    return _GLOBAL_DETECTOR
