"""
Sightengine API Client for Generative AI Image Detection.
Connects to Sightengine's enterprise genai detection endpoint (2,000 free checks/month).
Docs: https://sightengine.com/docs/ai-generated-image-detection
"""

import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("ai_detector.sightengine")

SIGHTENGINE_API_URL = "https://api.sightengine.com/1.0/check.json"


class SightengineClient:
    """
    Client for calling Sightengine's Generative AI Image Detection API.
    """

    def __init__(self, api_user: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_user = api_user or os.getenv("SIGHTENGINE_API_USER", "").strip()
        self.api_secret = api_secret or os.getenv("SIGHTENGINE_API_SECRET", "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_user and self.api_secret)

    def detect_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
        """
        Sends raw image bytes to Sightengine.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Sightengine API credentials (api_user / api_secret) are not configured.",
            }

        params = {
            "models": "genai",
            "api_user": self.api_user,
            "api_secret": self.api_secret,
        }
        files = {
            "media": (filename, image_bytes, "image/jpeg"),
        }

        try:
            response = requests.post(
                SIGHTENGINE_API_URL,
                data=params,
                files=files,
                timeout=25,
            )
            data = response.json()

            if response.status_code != 200 or data.get("status") != "success":
                error_msg = data.get("error", {}).get("message", "Sightengine API call failed")
                return {"success": False, "error": error_msg, "raw": data}

            ai_score = float(data.get("type", {}).get("ai_generated", 0.0))
            return {
                "success": True,
                "ai_score": round(ai_score, 4),
                "raw_response": data,
            }
        except Exception as e:
            return {"success": False, "error": f"Sightengine connection error: {str(e)}"}

    def detect_url(self, image_url: str) -> Dict[str, Any]:
        """
        Sends an image URL to Sightengine for remote analysis.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Sightengine API credentials (api_user / api_secret) are not configured.",
            }

        params = {
            "models": "genai",
            "url": image_url,
            "api_user": self.api_user,
            "api_secret": self.api_secret,
        }

        try:
            response = requests.post(
                SIGHTENGINE_API_URL,
                data=params,
                timeout=25,
            )
            data = response.json()

            if response.status_code != 200 or data.get("status") != "success":
                error_msg = data.get("error", {}).get("message", "Sightengine API call failed")
                return {"success": False, "error": error_msg, "raw": data}

            ai_score = float(data.get("type", {}).get("ai_generated", 0.0))
            return {
                "success": True,
                "ai_score": round(ai_score, 4),
                "raw_response": data,
            }
        except Exception as e:
            return {"success": False, "error": f"Sightengine connection error: {str(e)}"}
