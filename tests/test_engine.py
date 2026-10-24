"""
Unit tests for the AIImageDetector core engine.
"""

import os
import sys
import unittest
import base64
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.detector import AIImageDetector, get_detector

class TestAIImageDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector(force_reload=True)
        cls.sample_dir = os.path.join(BASE_DIR, "tests", "sample_images")
        cls.sketch_path = os.path.join(cls.sample_dir, "human_sketch.jpg")
        cls.pattern_path = os.path.join(cls.sample_dir, "synthetic_pattern.png")

    def test_engine_info(self):
        info = self.detector.get_info()
        self.assertEqual(info["model_name"], "dima806/ai_vs_real_image_detection")
        self.assertIn("REAL", info["labels"])
        self.assertIn("FAKE", info["labels"])
        self.assertEqual(info["model_source"], "local_weights")

    def test_predict_from_filepath(self):
        res = self.detector.predict(self.sketch_path)
        self.assertTrue(res["success"])
        self.assertIn("is_ai_generated", res)
        self.assertIn("artificial", res["scores"])
        self.assertIn("human", res["scores"])
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertGreater(res["inference_time_ms"], 0.0)
        self.assertIn("refund_fraud_analysis", res)

    def test_predict_from_pil(self):
        img = Image.open(self.pattern_path)
        res = self.detector.predict(img)
        self.assertTrue(res["success"])
        self.assertEqual(res["image_size"]["width"], 512)
        self.assertEqual(res["image_size"]["height"], 512)

    def test_predict_from_bytes(self):
        with open(self.sketch_path, "rb") as f:
            data = f.read()
        res = self.detector.predict(data)
        self.assertTrue(res["success"])

    def test_predict_from_base64(self):
        with open(self.sketch_path, "rb") as f:
            b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
        res = self.detector.predict(b64)
        self.assertTrue(res["success"])

    def test_predict_batch(self):
        results = self.detector.predict_batch([self.sketch_path, self.pattern_path])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["success"])

    def test_threshold_behavior(self):
        res_low = self.detector.predict(self.sketch_path, threshold=0.0)
        self.assertTrue(res_low["is_ai_generated"])

        res_high = self.detector.predict(self.sketch_path, threshold=1.0)
        self.assertFalse(res_high["is_ai_generated"])

    def test_corrupt_input_handling(self):
        res = self.detector.predict(b"not-an-image-data-just-random-text")
        self.assertFalse(res["success"])
        self.assertIn("error", res)

if __name__ == "__main__":
    unittest.main()
