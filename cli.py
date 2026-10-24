"""
Command-Line Interface (CLI) for AI Image Detector & Refund Fraud Verification.
Supports both Sightengine Cloud API and Local ViT Engine.
"""

import sys
import os
import argparse
import json
from dotenv import load_dotenv

load_dotenv()

# Safe console encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.detector import get_detector


def format_output(result: dict):
    if not result.get("success", False):
        print(f"[ERROR] {result.get('error')}")
        return

    verdict = result["verdict"]
    confidence = result["confidence_percentage"]
    ai_score = result["scores"]["artificial"] * 100
    human_score = result["scores"]["human"] * 100
    ms = result["inference_time_ms"]
    device = result["device"].upper()
    provider = result.get("provider", "Unknown")
    refund = result.get("refund_fraud_analysis", {})

    badge = "[AI-GENERATED]" if result["is_ai_generated"] else "[GENUINE / REAL]"
    
    print("\n" + "=" * 58)
    print(f"Provider:             {provider}")
    print(f"Verdict:              {badge} {verdict}")
    print(f"Confidence:           {confidence}")
    print(f"AI / Fake Score:      {ai_score:.2f}%")
    print(f"Real / Authentic:     {human_score:.2f}%")
    print(f"Decision Threshold:   {result['threshold']}")
    if result.get("image_size"):
        print(f"Image Resolution:     {result['image_size']['width']}x{result['image_size']['height']} px")
    print(f"Inference Latency:    {ms:.2f} ms")
    print(f"Model:                {result['model']}")
    print(f"Execution Target:     {device}")
    
    if refund:
        print("-" * 58)
        print("RETURN / REFUND CLAIM ANALYSIS:")
        print(f"  Risk Level:         {refund.get('risk_level')}")
        print(f"  Recommended Action: {refund.get('recommended_action')}")
        print(f"  Camera EXIF Found:  {refund.get('has_camera_exif')}")
        print(f"  Policy Notes:       {refund.get('notes')}")
    print("=" * 58 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Detect whether an image is AI-generated and assess refund fraud risk.")
    parser.add_argument("image", nargs="?", help="Path to image file to analyze")
    parser.add_argument("--url", help="Direct URL to an image to analyze")
    parser.add_argument("--provider", choices=["auto", "sightengine", "local"], default="auto", help="Detection provider (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default 0.5)")
    parser.add_argument("--sightengine-user", help="Sightengine API User ID")
    parser.add_argument("--sightengine-secret", help="Sightengine API Secret")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    if not args.image and not args.url:
        parser.print_help()
        sys.exit(1)

    detector = get_detector(
        sightengine_user=args.sightengine_user,
        sightengine_secret=args.sightengine_secret,
        preferred_provider=args.provider,
    )

    target = args.url if args.url else args.image
    res = detector.predict(
        target,
        threshold=args.threshold,
        provider=args.provider,
        sightengine_user=args.sightengine_user,
        sightengine_secret=args.sightengine_secret,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        format_output(res)


if __name__ == "__main__":
    main()
