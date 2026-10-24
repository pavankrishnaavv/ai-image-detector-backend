"""
Script to save/export the 'dima806/ai_vs_real_image_detection' model and preprocessor
to a local directory (./model_weights) so the backend can run fully offline
and self-contained.
"""

import os
from transformers import pipeline

MODEL_NAME = "dima806/ai_vs_real_image_detection"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")

def export_model():
    print(f"Loading '{MODEL_NAME}' pipeline...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pipe = pipeline("image-classification", model=MODEL_NAME)
    print(f"Saving to '{OUTPUT_DIR}'...")
    pipe.save_pretrained(OUTPUT_DIR)
    
    print("✅ Model weights and preprocessor configuration successfully saved to:", OUTPUT_DIR)
    print("Files in model_weights:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_model()
