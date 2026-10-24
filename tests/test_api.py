"""
API endpoint integration tests for FastAPI backend.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_dir = os.path.join(BASE_DIR, "tests", "sample_images")
        cls.sample_path = os.path.join(cls.sample_dir, "human_sketch.jpg")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_info_endpoint(self):
        response = self.client.get("/api/v1/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "umm-maybe/AI-image-detector")

    def test_detect_upload(self):
        with open(self.sample_path, "rb") as f:
            response = self.client.post(
                "/api/v1/detect?threshold=0.5",
                files={"file": ("sketch.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("is_ai_generated", data)
        self.assertIn("verdict", data)
        self.assertIn("confidence_percentage", data)

    def test_detect_base64(self):
        import base64
        with open(self.sample_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.post(
            "/api/v1/detect/base64",
            json={"image_base64": b64_str, "threshold": 0.5}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    def test_detect_batch_upload(self):
        with open(self.sample_path, "rb") as f1, open(self.sample_path, "rb") as f2:
            response = self.client.post(
                "/api/v1/detect/batch",
                files=[
                    ("files", ("image1.jpg", f1, "image/jpeg")),
                    ("files", ("image2.jpg", f2, "image/jpeg")),
                ]
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_images"], 2)
        self.assertEqual(len(data["results"]), 2)

if __name__ == "__main__":
    unittest.main()
