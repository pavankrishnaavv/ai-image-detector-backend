"""
Command-Line Interface (CLI) for AI Image Detector & Refund Fraud Verification.
Quickly test local files or remote URLs.
"""

import sys
import os
import argparse
import json

# Safe console encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure backend root is in python path
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
    refund = result.get("refund_fraud_analysis", {})

    badge = "[AI-GENERATED]" if result["is_ai_generated"] else "[GENUINE / REAL]"
    
    print("\n" + "=" * 55)
    print(f"Verdict:              {badge} {verdict}")
    print(f"Confidence:           {confidence}")
    print(f"AI/Fake Probability:  {ai_score:.2f}%")
    print(f"Real/Authentic Score: {human_score:.2f}%")
    print(f"Decision Threshold:   {result['threshold']}")
    if result.get("image_size"):
        print(f"Image Resolution:     {result['image_size']['width']}x{result['image_size']['height']} px")
    print(f"Inference Latency:    {ms:.2f} ms")
    print(f"Model Architecture:   {result['model']}")
    print(f"Compute Device:       {device}")
    
    if refund:
        print("-" * 55)
        print("RETURN / REFUND CLAIM ANALYSIS:")
        print(f"  Risk Level:         {refund.get('risk_level')}")
        print(f"  Recommended Action: {refund.get('recommended_action')}")
        print(f"  Camera EXIF Found:  {refund.get('has_camera_exif')}")
        print(f"  Policy Notes:       {refund.get('notes')}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Detect whether an image is AI-generated and assess refund fraud risk.")
    parser.add_argument("image", nargs="?", help="Path to image file to analyze")
    parser.add_argument("--url", help="Direct URL to an image to analyze")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default 0.5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    parser.add_argument("--batch", nargs="+", help="Multiple image paths for batch classification")

    args = parser.parse_args()

    if not args.image and not args.url and not args.batch:
        parser.print_help()
        sys.exit(1)

    print("[INFO] Initializing detector engine...")
    detector = get_detector(force_reload=True)

    if args.batch:
        print(f"[INFO] Analyzing batch of {len(args.batch)} images...")
        results = detector.predict_batch(args.batch, threshold=args.threshold)
        for path, res in zip(args.batch, results):
            print(f"\n--- File: {path} ---")
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                format_output(res)
        return

    target = args.url if args.url else args.image
    res = detector.predict(target, threshold=args.threshold)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        format_output(res)


if __name__ == "__main__":
    main()
