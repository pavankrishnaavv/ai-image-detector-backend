"""
Unit tests for the DigitalForensicsAnalyzer engine.
"""

import os
import sys
import unittest
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.forensics import DigitalForensicsAnalyzer


class TestDigitalForensicsAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = DigitalForensicsAnalyzer()
        cls.sample_dir = os.path.join(BASE_DIR, "tests", "sample_images")
        cls.sketch_path = os.path.join(cls.sample_dir, "human_sketch.jpg")
        cls.pattern_path = os.path.join(cls.sample_dir, "synthetic_pattern.png")

    def test_metadata_audit(self):
        img = Image.open(self.sketch_path)
        meta = self.analyzer.inspect_metadata(img)
        self.assertIn("has_camera_hardware_exif", meta)
        self.assertIn("metadata_risk_score", meta)
        self.assertIn("summary", meta)
        self.assertIsInstance(meta["provenance_markers"], list)

    def test_metadata_synthetic_detection(self):
        # Fake raw bytes containing C2PA marker
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        fake_bytes = b"header_data_c2pa_manifest_trainedAlgorithmicMedia_tail"
        meta = self.analyzer.inspect_metadata(img, raw_bytes=fake_bytes)
        self.assertTrue(meta["c2pa_provenance_detected"])
        self.assertGreaterEqual(meta["metadata_risk_score"], 0.9)

    def test_fft_analysis(self):
        img = Image.open(self.sketch_path)
        fft_res = self.analyzer.analyze_fft(img)
        self.assertIn("high_freq_peaks_count", fft_res)
        self.assertIn("spectral_energy_ratio", fft_res)
        self.assertIn("checkerboard_detected", fft_res)
        self.assertIn("spectrum_preview_b64", fft_res)
        self.assertTrue(fft_res["spectrum_preview_b64"].startswith("data:image/jpeg;base64,"))

    def test_ela_analysis(self):
        img = Image.open(self.sketch_path)
        ela_res = self.analyzer.analyze_ela(img)
        self.assertIn("mean_error", ela_res)
        self.assertIn("max_error", ela_res)
        self.assertIn("tampering_detected", ela_res)
        self.assertIn("heatmap_preview_b64", ela_res)
        self.assertTrue(ela_res["heatmap_preview_b64"].startswith("data:image/jpeg;base64,"))

    def test_noise_residuals(self):
        img = Image.open(self.sketch_path)
        noise_res = self.analyzer.analyze_noise_residuals(img)
        self.assertIn("total_noise_variance", noise_res)
        self.assertIn("noise_inconsistency_index", noise_res)
        self.assertIn("unnatural_smoothing_detected", noise_res)

    def test_full_forensics(self):
        img = Image.open(self.sketch_path)
        full_res = self.analyzer.run_full_forensics(img)
        self.assertIn("composite_forensic_score", full_res)
        self.assertIn("forensic_verdict", full_res)
        self.assertIn("key_indicators", full_res)
        self.assertGreaterEqual(len(full_res["key_indicators"]), 1)
        self.assertIn("metadata_audit", full_res)
        self.assertIn("frequency_analysis_fft", full_res)
        self.assertIn("error_level_analysis_ela", full_res)
        self.assertIn("noise_residuals", full_res)


if __name__ == "__main__":
    unittest.main()
