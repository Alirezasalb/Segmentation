# test_sam_cli.py
import numpy as np
from PIL import Image
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

# Paths
MODEL_CKPT = "sam_vit_b_01ec64.pth"
INPUT_IMAGE = "11.jpg"
OUTPUT_MASK = "outpt_mask.png"
OUTPUT_OVERLAY = "outpt_overlay.jpg"

# Verify files exist
assert os.path.exists(MODEL_CKPT), f"❌ Model not found: {MODEL_CKPT}"
assert os.path.exists(INPUT_IMAGE), f"❌ Input image not found: {INPUT_IMAGE}"

print("✅ Loading image...")
image = Image.open(INPUT_IMAGE).convert("RGB")
image_np = np.array(image)
print(f"  → Image size: {image_np.shape}")

print("✅ Loading SAM model...")
predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=MODEL_CKPT))
predictor.set_image(image_np)
print("  → SAM ready!")

# Define point prompts (you can change these!)
# Format: [[x1, y1], [x2, y2], ...]
# Label: 1 = foreground, 0 = background
input_points = np.array([[200, 200]])  # ← CHANGE THESE COORDINATES!
input_labels = np.array([1])           # 1 = object we want

print(f"✅ Segmenting with points: {input_points.tolist()}")
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# Pick best mask
best_idx = np.argmax(scores)
mask = masks[best_idx]
print(f"  → Best mask score: {scores[best_idx]:.3f}")
print(f"  → Mask shape: {mask.shape}, non-zero pixels: {mask.sum()}")

# Save mask
mask_image = Image.fromarray((mask * 255).astype(np.uint8))
mask_image.save(OUTPUT_MASK)
print(f"✅ Mask saved to: {OUTPUT_MASK}")

# Create and save overlay
overlay = image_np.copy()
colored_mask = np.zeros_like(overlay)
colored_mask[:, :, 2] = mask * 255  # Blue channel
overlay = cv2.addWeighted(colored_mask, 0.6, overlay, 0.4, 0)
cv2.imwrite(OUTPUT_OVERLAY, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"✅ Overlay saved to: {OUTPUT_OVERLAY}")

print("\n🎉 Done! Check your folder for output images.")