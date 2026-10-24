"""
Generates local test images for unit testing the AI detector.
"""

import os
from PIL import Image, ImageDraw

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
os.makedirs(SAMPLE_DIR, exist_ok=True)

def create_samples():
    # 1. Plain photographic gradient / landscape approximation
    img1 = Image.new("RGB", (400, 300), color=(135, 206, 235))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([0, 150, 400, 300], fill=(34, 139, 34)) # Grass
    draw.ellipse([50, 50, 110, 110], fill=(255, 223, 0))   # Sun
    img1.save(os.path.join(SAMPLE_DIR, "human_sketch.jpg"), quality=95)

    # 2. Geometric multi-color synthetic art
    img2 = Image.new("RGB", (512, 512), color=(20, 20, 30))
    draw2 = ImageDraw.Draw(img2)
    for i in range(10, 250, 15):
        draw2.rectangle([i, i, 512-i, 512-i], outline=(i % 255, (i*2)%255, (255-i)), width=2)
    img2.save(os.path.join(SAMPLE_DIR, "synthetic_pattern.png"))

    print(f"Created sample images in {SAMPLE_DIR}")

if __name__ == "__main__":
    create_samples()
