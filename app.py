# app.py
import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2
import os
import matplotlib.pyplot as plt
import traceback

# Safety check
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    st.error("❌ Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
    st.stop()

CHECKPOINT_PATH = os.path.join(os.getcwd(), "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"❌ Model not found: {CHECKPOINT_PATH}")
    st.stop()


@st.cache_resource
def load_sam():
    st.info("🧠 Loading SAM model (one-time)...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    st.success("✅ SAM loaded!")
    return SamPredictor(sam)


def overlay_mask(image, mask, alpha=0.6):
    image = np.array(image)
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:, :, 2] = (mask > 0) * 255
    cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


# Initialize
for key in ["image", "points", "labels", "mask", "x", "y"]:
    if key not in st.session_state:
        if key == "points" or key == "labels":
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# Sidebar
with st.sidebar:
    uploaded = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        st.session_state.image = img
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.mask = None
        st.session_state.x = img.width // 2
        st.session_state.y = img.height // 2

st.title("✂️ SAM Universal Segmenter (Debug Mode)")

if st.session_state.image is None:
    st.info("👆 Upload an image to begin.")
else:
    w, h = st.session_state.image.size

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Foreground"):
            st.session_state.current_label = 1
    with col2:
        if st.button("🔴 Background"):
            st.session_state.current_label = 0

    st.session_state.x = st.number_input("X", 0, w - 1, st.session_state.x)
    st.session_state.y = st.number_input("Y", 0, h - 1, st.session_state.y)

    if st.button("➕ ADD POINT"):
        if "current_label" not in st.session_state:
            st.warning("⚠️ Select point type first!")
        else:
            st.session_state.points.append([st.session_state.x, st.session_state.y])
            st.session_state.labels.append(st.session_state.current_label)
            st.success(f"✅ Added point at ({st.session_state.x}, {st.session_state.y})")

    # Show points on image
    if st.session_state.points:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(st.session_state.image)
        pts = np.array(st.session_state.points)
        lbls = np.array(st.session_state.labels)
        ax.scatter(pts[lbls == 1, 0], pts[lbls == 1, 1], c='green', marker='*', s=200, edgecolor='white')
        ax.scatter(pts[lbls == 0, 0], pts[lbls == 0, 1], c='red', marker='*', s=200, edgecolor='white')
        ax.axis('off')
        st.pyplot(fig)

    # SEGMENT BUTTON
    if st.button("✂️ RUN SEGMENTATION"):
        st.write("DEBUG: Button clicked!")
        if not st.session_state.points:
            st.warning("⚠️ Add points first!")
        else:
            st.write("DEBUG: Points exist, starting SAM...")
            try:
                with st.spinner("🧠 Running SAM segmentation..."):
                    predictor = load_sam()
                    image_np = np.array(st.session_state.image)
                    predictor.set_image(image_np)

                    coords = np.array(st.session_state.points)
                    labels = np.array(st.session_state.labels)

                    masks, scores, logits = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=True,
                    )

                    best_idx = int(np.argmax(scores))
                    mask = masks[best_idx]

                    # DEBUG: print mask stats
                    st.write(f"🔍 Mask shape: {mask.shape}, dtype: {mask.dtype}")
                    st.write(f"📊 Mask sum (pixels): {mask.sum()}, min: {mask.min()}, max: {mask.max()}")

                    if mask.sum() == 0:
                        st.warning("⚠️ Warning: Generated mask is EMPTY (all zeros). Try different points!")
                    else:
                        st.session_state.mask = mask
                        st.success("✅ Segmentation successful!")

            except Exception as e:
                st.error(f"💥 Error during segmentation:")
                st.code(traceback.format_exc())

    # SHOW RESULT
    if st.session_state.mask is not None:
        st.subheader("🎨 Result")
        overlay = overlay_mask(st.session_state.image, st.session_state.mask)
        st.image(overlay, caption="Segmentation Overlay", use_column_width=True)

        # Also show raw mask
        st.image(st.session_state.mask.astype(np.float32), caption="Raw Mask (white = segmented)",
                 use_column_width=True)

        mask_pil = Image.fromarray((st.session_state.mask * 255).astype(np.uint8))
        st.download_button("📥 Download Mask", mask_pil.tobytes(), "sam_mask.png", "image/png")