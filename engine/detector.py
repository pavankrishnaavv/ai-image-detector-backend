"""
Core AI Image Detector Engine.
Wraps the 'umm-maybe/AI-image-detector' Swin Transformer model.
Supports single and batch image classification with flexible inputs:
- PIL Image
- Raw bytes
- Base64 string
- Remote URL
- Local file path
"""

import os
import io
import time
import base64
import logging
from typing import List, Dict, Any, Union, Optional
from PIL import Image, UnidentifiedImageError
import torch
from transformers import pipeline

logger = logging.getLogger("ai_detector.engine")

DEFAULT_MODEL_NAME = "umm-maybe/AI-image-detector"
LOCAL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_weights"
)


class AIImageDetector:
    """
    Inference engine for AI Image Detection using Hugging Face's
    umm-maybe/AI-image-detector model.
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
        :param threshold: Classification threshold for 'artificial' label (default 0.5).
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

    def _normalize_image(self, image_input: Union[Image.Image, bytes, str]) -> Image.Image:
        """
        Converts any supported input format into a RGB PIL.Image.
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
                # Remote URL fetch
                import urllib.request
                req = urllib.request.Request(
                    image_input,
                    headers={"User-Agent": "AI-Image-Detector-Backend/1.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    img_data = response.read()
                img = Image.open(io.BytesIO(img_data))
            elif os.path.isfile(image_input):
                img = Image.open(image_input)
            else:
                # Attempt raw base64 string
                try:
                    img_data = base64.b64decode(image_input)
                    img = Image.open(io.BytesIO(img_data))
                except Exception:
                    raise ValueError(
                        f"Unsupported image string input. Must be a valid file path, URL, or base64 data URI."
                    )
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB mode (handles RGBA, Palette, Grayscale, CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform detection on a single image.
        :param image_input: PIL Image, bytes, base64 string, URL, or filepath.
        :param threshold: Custom decision threshold (defaults to self.threshold).
        :return: Structured prediction dictionary.
        """
        eff_threshold = threshold if threshold is not None else self.threshold
        start_time = time.perf_counter()

        try:
            image = self._normalize_image(image_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process image: {str(e)}",
                "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Run model inference
        raw_results = self._pipeline(image)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Parse classification scores
        scores = {}
        for item in raw_results:
            scores[item["label"].lower()] = float(item["score"])

        artificial_score = scores.get("artificial", 0.0)
        human_score = scores.get("human", 0.0)

        # In case scores don't sum to 1 (softmax usually does)
        total = artificial_score + human_score
        if total > 0:
            artificial_prob = round(artificial_score / total, 4)
            human_prob = round(human_score / total, 4)
        else:
            artificial_prob = round(artificial_score, 4)
            human_prob = round(human_score, 4)

        is_ai = artificial_prob >= eff_threshold
        verdict = "AI-Generated" if is_ai else "Human-Created"
        confidence = artificial_prob if is_ai else human_prob

        return {
            "success": True,
            "is_ai_generated": is_ai,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "scores": {
                "artificial": artificial_prob,
                "human": human_prob,
            },
            "threshold": eff_threshold,
            "image_size": {
                "width": image.width,
                "height": image.height,
            },
            "inference_time_ms": inference_time_ms,
            "model": "umm-maybe/AI-image-detector",
            "device": self.device,
        }

    def predict_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform detection on a batch of images.
        :param images: List of image inputs.
        :param threshold: Custom decision threshold.
        :return: List of prediction dictionaries.
        """
        results = []
        for img in images:
            results.append(self.predict(img, threshold=threshold))
        return results

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the active engine and model."""
        return {
            "model_name": "umm-maybe/AI-image-detector",
            "architecture": "SwinForImageClassification",
            "labels": ["artificial", "human"],
            "device": self.device,
            "model_source": "local_weights" if self.model_path == LOCAL_WEIGHTS_PATH else "huggingface",
            "default_threshold": self.threshold,
        }


# Global singleton instance
_GLOBAL_DETECTOR: Optional[AIImageDetector] = None


def get_detector(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> AIImageDetector:
    """
    Get or create the global singleton detector instance.
    """
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = AIImageDetector(
            model_name_or_path=model_name_or_path,
            device=device,
            threshold=threshold,
        )
    return _GLOBAL_DETECTOR
