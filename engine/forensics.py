"""
Digital Forensics & Neural Pattern Analyzer.
Analyzes image metadata (EXIF/C2PA/IPTC), 2D Fourier (FFT) grid artifacts,
Error Level Analysis (ELA) compression deltas, and noise residual consistency.
"""

import os
import io
import re
import base64
import logging
from typing import Dict, Any, Optional, List
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np
from scipy.ndimage import laplace

logger = logging.getLogger("ai_detector.forensics")

KNOWN_AI_SIGNATURES = [
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "stablediffusion",
    "comfyui",
    "automatic1111",
    "novelai",
    "flux",
    "imagen",
    "bing image creator",
    "adobe firefly",
    "dreamstudio",
    "photoleap",
    "wombo",
    "craiyon",
]

C2PA_MARKERS = [
    b"c2pa",
    b"C2PA",
    b"http://c2pa.org",
    b"trainedAlgorithmicMedia",
    b"contentCredentials",
    b"adobe:contentcredentials",
]


class DigitalForensicsAnalyzer:
    """
    Forensics engine detecting synthetic generative traces:
    1. Metadata & Hardware Provenance (EXIF, C2PA, IPTC, AI tool signatures)
    2. 2D FFT Frequency Domain Analysis (Checkerboard & upsampling artifacts)
    3. Error Level Analysis (ELA) (Compression quantization deltas & inpainting)
    4. Sensor Noise Residuals (Laplacian variance & diffusion smoothing consistency)
    """

    def __init__(self, ela_quality: int = 90):
        self.ela_quality = ela_quality

    def inspect_metadata(
        self, img: Image.Image, raw_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Audits image metadata, camera hardware tags, C2PA provenance, and AI tool signatures."""
        metadata = {
            "has_camera_hardware_exif": False,
            "camera_make": None,
            "camera_model": None,
            "lens_model": None,
            "focal_length": None,
            "iso": None,
            "exposure_time": None,
            "f_number": None,
            "software": None,
            "date_time": None,
            "ai_software_signature_detected": False,
            "detected_ai_software": None,
            "c2pa_provenance_detected": False,
            "provenance_markers": [],
            "all_tags_count": 0,
            "metadata_risk_score": 0.5,  # neutral default
            "summary": "",
        }

        # 1. EXIF Inspection
        try:
            exif = img.getexif()
            if exif:
                metadata["all_tags_count"] = len(exif)
                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    str_val = str(value).strip()

                    if tag_name == "Make":
                        metadata["camera_make"] = str_val
                    elif tag_name == "Model":
                        metadata["camera_model"] = str_val
                    elif tag_name == "LensModel":
                        metadata["lens_model"] = str_val
                    elif tag_name == "FocalLength":
                        metadata["focal_length"] = str_val
                    elif tag_name == "ISOSpeedRatings":
                        try:
                            metadata["iso"] = int(value)
                        except Exception:
                            metadata["iso"] = str_val
                    elif tag_name == "ExposureTime":
                        metadata["exposure_time"] = str_val
                    elif tag_name == "FNumber":
                        try:
                            metadata["f_number"] = float(value)
                        except Exception:
                            metadata["f_number"] = str_val
                    elif tag_name in ("DateTime", "DateTimeOriginal"):
                        metadata["date_time"] = str_val
                    elif tag_name == "Software":
                        metadata["software"] = str_val

                    # Check for AI tool signatures in tag values
                    val_lower = str_val.lower()
                    for sig in KNOWN_AI_SIGNATURES:
                        if sig in val_lower:
                            metadata["ai_software_signature_detected"] = True
                            metadata["detected_ai_software"] = sig.title()

                if metadata["camera_make"] or metadata["camera_model"]:
                    metadata["has_camera_hardware_exif"] = True
        except Exception as e:
            logger.warning(f"EXIF parsing error: {e}")

        # 2. Raw Byte Provenance & C2PA scan
        if raw_bytes:
            for marker in C2PA_MARKERS:
                if marker in raw_bytes:
                    metadata["c2pa_provenance_detected"] = True
                    marker_str = marker.decode("utf-8", errors="ignore")
                    if marker_str not in metadata["provenance_markers"]:
                        metadata["provenance_markers"].append(marker_str)

        # 3. Calculate Metadata Risk Score & Summary
        if metadata["ai_software_signature_detected"] or metadata["c2pa_provenance_detected"]:
            metadata["metadata_risk_score"] = 0.98
            reasons = []
            if metadata["ai_software_signature_detected"]:
                reasons.append(f"AI software header '{metadata['detected_ai_software']}' detected")
            if metadata["c2pa_provenance_detected"]:
                reasons.append(f"C2PA synthetic provenance markers {metadata['provenance_markers']} detected")
            metadata["summary"] = "Definitive synthetic media markers found in image metadata: " + "; ".join(reasons)
        elif metadata["has_camera_hardware_exif"]:
            # Real camera hardware profile
            metadata["metadata_risk_score"] = 0.15
            cam = f"{metadata['camera_make'] or ''} {metadata['camera_model'] or ''}".strip()
            metadata["summary"] = f"Genuine camera hardware EXIF tags detected ({cam}). Consistent with authentic capture."
        else:
            # Metadata stripped or absent (common in AI generations or web-scraped images)
            metadata["metadata_risk_score"] = 0.65
            metadata["summary"] = "No physical camera hardware EXIF tags found. Metadata is stripped or missing, typical of AI generation or web downloads."

        return metadata

    def analyze_ela(self, img: Image.Image) -> Dict[str, Any]:
        """
        Error Level Analysis (ELA).
        Re-saves at fixed JPEG quality and inspects quantization delta patterns.
        """
        rgb = img.convert("RGB")
        w, h = rgb.size

        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=self.ela_quality)
        buf.seek(0)
        recompressed = Image.open(buf)

        diff = ImageChops.difference(rgb, recompressed)
        diff_arr = np.asarray(diff, dtype=np.float32)

        mean_err = float(np.mean(diff_arr))
        max_err = float(np.max(diff_arr))
        std_err = float(np.std(diff_arr))

        # Check patch consistency (tampering / inpainting localization)
        patch_size = 32
        patch_means = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = diff_arr[i : i + patch_size, j : j + patch_size]
                patch_means.append(np.mean(patch))

        patch_means = np.array(patch_means) if patch_means else np.array([mean_err])
        patch_inconsistency = float(np.std(patch_means) / (np.mean(patch_means) + 1e-6))
        tampering_detected = patch_inconsistency > 1.2 or max_err > 25.0

        # ELA anomaly score (0.0 to 1.0)
        # Spliced / AI inpainted artifacts typically push max_err > 15 and patch inconsistency > 1.0
        ela_score = min(1.0, max(0.0, (mean_err * 0.15) + (patch_inconsistency * 0.35) + (max_err / 50.0) * 0.5))

        # Generate amplified ELA visualization heatmap
        scale = 255.0 / max(max_err, 1.0)
        ela_enhanced = ImageEnhance.Brightness(diff).enhance(scale * 0.85)

        thumb_w = min(w, 400)
        thumb_h = max(int(h * (thumb_w / w)), 1)
        thumb = ela_enhanced.resize((thumb_w, thumb_h))
        out_buf = io.BytesIO()
        thumb.save(out_buf, "JPEG", quality=85)
        ela_b64 = "data:image/jpeg;base64," + base64.b64encode(out_buf.getvalue()).decode("ascii")

        interpretation = (
            "Localized compression delta anomalies detected, indicating potential synthetic inpainting or editing."
            if tampering_detected
            else "Uniform error distribution across compression blocks, consistent with single-pass capture."
        )

        return {
            "mean_error": round(mean_err, 3),
            "max_error": round(max_err, 2),
            "std_error": round(std_err, 3),
            "patch_inconsistency": round(patch_inconsistency, 3),
            "tampering_detected": tampering_detected,
            "ela_score": round(ela_score, 4),
            "interpretation": interpretation,
            "heatmap_preview_b64": ela_b64,
        }

    def analyze_fft(self, img: Image.Image) -> Dict[str, Any]:
        """
        2D Fast Fourier Transform (FFT) Frequency Analysis.
        Detects periodic checkerboard grid artifacts and high-frequency energy anomalies
        characteristic of transposed convolutions and neural upsamplers.
        """
        gray = img.convert("L")
        w, h = gray.size
        arr = np.asarray(gray, dtype=np.float32)

        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-6)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = min(cx, cy)

        hf_mask = (r > (max_r * 0.5)) & (r <= max_r)
        lf_mask = r <= (max_r * 0.25)

        hf_energy = float(np.mean(np.abs(fshift)[hf_mask]))
        lf_energy = float(np.mean(np.abs(fshift)[lf_mask]))
        spectral_ratio = float(hf_energy / (lf_energy + 1e-8))

        hf_spectrum = mag[hf_mask]
        hf_mean = float(np.mean(hf_spectrum))
        hf_std = float(np.std(hf_spectrum))

        # Count isolated periodic spikes (> 3.0 std dev above ring mean)
        spikes = int(np.sum(hf_spectrum > (hf_mean + 3.0 * hf_std)))
        peak_density = float(spikes / max(len(hf_spectrum), 1))

        # Periodic checkerboard grid artifact detection
        checkerboard_detected = spikes >= 35 or peak_density > 0.00015

        # FFT grid anomaly score (0.0 to 1.0)
        grid_score = min(1.0, max(0.0, (spikes / 100.0) * 0.6 + (peak_density * 2000.0) * 0.4))

        # Generate FFT 2D spectrum preview image
        norm_mag = ((mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6) * 255).astype(np.uint8)
        fft_pil = Image.fromarray(norm_mag)
        fft_thumb_buf = io.BytesIO()
        fft_pil.resize((256, 256)).save(fft_thumb_buf, "JPEG", quality=85)
        fft_b64 = "data:image/jpeg;base64," + base64.b64encode(fft_thumb_buf.getvalue()).decode("ascii")

        interpretation = (
            f"Periodic high-frequency spectral spikes detected ({spikes} peaks), characteristic of neural upsampling / checkerboard residuals."
            if checkerboard_detected
            else "Natural smooth power spectrum decay, typical of optical camera sensors."
        )

        return {
            "high_freq_peaks_count": spikes,
            "peak_density": round(peak_density, 6),
            "spectral_energy_ratio": round(spectral_ratio, 6),
            "checkerboard_detected": checkerboard_detected,
            "fft_anomaly_score": round(grid_score, 4),
            "interpretation": interpretation,
            "spectrum_preview_b64": fft_b64,
        }

    def analyze_noise_residuals(self, img: Image.Image) -> Dict[str, Any]:
        """
        Noise Residual & Laplacian Variance (Sensor PRNU & Smoothing Analysis).
        Evaluates physical photon noise vs synthetic diffusion denoising smoothing.
        """
        gray = img.convert("L")
        w, h = gray.size
        arr = np.asarray(gray, dtype=np.float32)

        lap = laplace(arr)
        total_var = float(np.var(lap))

        patch_size = 32
        p_vars = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                p = lap[i : i + patch_size, j : j + patch_size]
                p_vars.append(np.var(p))

        p_vars = np.array(p_vars) if p_vars else np.array([total_var])
        mean_p_var = float(np.mean(p_vars))
        std_p_var = float(np.std(p_vars))
        noise_inconsistency = float(std_p_var / (mean_p_var + 1e-6))

        # Diffusion models display unnatural patch variance discrepancies
        unnatural_smoothing = noise_inconsistency > 1.05

        noise_score = min(1.0, max(0.0, (noise_inconsistency - 0.7) * 1.5)) if noise_inconsistency > 0.7 else 0.1

        interpretation = (
            f"Spatial noise inconsistency ({noise_inconsistency:.2f}) indicates uneven synthetic smoothing typical of diffusion denoising steps."
            if unnatural_smoothing
            else "Uniform spatial noise distribution consistent with physical sensor photon noise."
        )

        return {
            "total_noise_variance": round(total_var, 2),
            "noise_inconsistency_index": round(noise_inconsistency, 3),
            "unnatural_smoothing_detected": unnatural_smoothing,
            "noise_anomaly_score": round(noise_score, 4),
            "interpretation": interpretation,
        }

    def run_full_forensics(
        self, img: Image.Image, raw_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Runs the complete suite of digital forensics tests."""
        meta = self.inspect_metadata(img, raw_bytes=raw_bytes)
        ela = self.analyze_ela(img)
        fft = self.analyze_fft(img)
        noise = self.analyze_noise_residuals(img)

        # Composite Forensic AI Probability
        weights = {
            "metadata": 0.25,
            "fft": 0.35,
            "ela": 0.20,
            "noise": 0.20,
        }
        composite_score = (
            meta["metadata_risk_score"] * weights["metadata"]
            + fft["fft_anomaly_score"] * weights["fft"]
            + ela["ela_score"] * weights["ela"]
            + noise["noise_anomaly_score"] * weights["noise"]
        )
        composite_score = round(min(1.0, max(0.0, composite_score)), 4)

        # Build Explainable Indicators
        indicators = []
        if meta["ai_software_signature_detected"] or meta["c2pa_provenance_detected"]:
            indicators.append(f"Metadata: {meta['summary']}")
        elif not meta["has_camera_hardware_exif"]:
            indicators.append("Metadata: Camera sensor hardware EXIF is missing/stripped.")
        else:
            indicators.append(f"Metadata: Verified camera capture ({meta['camera_make']} {meta['camera_model']}).")

        if fft["checkerboard_detected"]:
            indicators.append(f"FFT 2D Spectrum: {fft['high_freq_peaks_count']} periodic grid spikes detected (neural upsampling artifact).")
        else:
            indicators.append("FFT 2D Spectrum: Natural continuous radial frequency decay.")

        if ela["tampering_detected"]:
            indicators.append(f"Error Level Analysis: Localized compression delta anomalies (max delta {ela['max_error']}).")

        if noise["unnatural_smoothing_detected"]:
            indicators.append(f"Noise Residuals: High noise inconsistency index ({noise['noise_inconsistency_index']}), characteristic of diffusion denoising.")

        if composite_score >= 0.65:
            verdict = "HIGH_SYNTHETIC_PROBABILITY"
        elif composite_score >= 0.40:
            verdict = "SUSPICIOUS_ANOMALIES"
        else:
            verdict = "NATURAL_SENSOR_CONSISTENT"

        return {
            "composite_forensic_score": composite_score,
            "forensic_verdict": verdict,
            "key_indicators": indicators,
            "metadata_audit": meta,
            "frequency_analysis_fft": fft,
            "error_level_analysis_ela": ela,
            "noise_residuals": noise,
        }
