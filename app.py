import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import time

# Safety checks
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    st.error("❌ Please install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
    st.stop()

if not os.path.exists("sam_vit_b_01ec64.pth"):
    st.error("❌ Model file 'sam_vit_b_01ec64.pth' not found in current directory!")
    st.markdown("👉 Download it from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
    st.stop()


@st.cache_resource
def load_sam_predictor():
    """Load SAM model once and reuse"""
    return SamPredictor(sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth"))


def resize_image(image, max_size=768):
    """Resize large images to speed up CPU inference"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        return image.resize((new_w, new_h))
    return image


def create_overlay(image, mask, alpha=0.6):
    """Create overlay of mask on image"""
    image_np = np.array(image)
    overlay = image_np.copy()
    colored_mask = np.zeros_like(image_np)
    colored_mask[:, :, 2] = mask * 255  # Blue channel
    return cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)


# Initialize session state
if "image" not in st.session_state:
    st.session_state.image = None
if "mask" not in st.session_state:
    st.session_state.mask = None
if "points" not in st.session_state:
    st.session_state.points = []
if "labels" not in st.session_state:
    st.session_state.labels = []

# UI
st.set_page_config(page_title="SAM Universal Segmenter", layout="wide")
st.title("✂️ SAM Universal Segmenter")
st.caption("Powered by Meta's Segment Anything Model (vit_b)")

# Sidebar
with st.sidebar:
    st.header("📤 Upload & Settings")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.image = resize_image(image)
        st.session_state.mask = None
        st.session_state.points = []
        st.session_state.labels = []
        st.success(f"✅ Image loaded ({st.session_state.image.size[0]}×{st.session_state.image.size[1]})")

# Main area
if st.session_state.image is None:
    st.info("👆 Upload an image to get started!")
    st.markdown("💡 **Tip**: For best results, click on the object you want to segment.")
else:
    # Display image
    st.image(st.session_state.image, caption="Input Image", use_column_width=True)

    # Point input
    st.subheader("📍 Add Points")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Foreground (Object)"):
            st.session_state.current_label = 1
    with col2:
        if st.button("🔴 Background"):
            st.session_state.current_label = 0

    w, h = st.session_state.image.size
    x = st.number_input("X coordinate", 0, w - 1, w // 2)
    y = st.number_input("Y coordinate", 0, h - 1, h // 2)

    if st.button("➕ Add Point"):
        if "current_label" not in st.session_state:
            st.warning("⚠️ Please select foreground or background first!")
        else:
            st.session_state.points.append([x, y])
            st.session_state.labels.append(st.session_state.current_label)
            st.success(f"✅ Point added at ({x}, {y})")

    # Show current points
    if st.session_state.points:
        st.write("**Current Points:**")
        for i, (pt, lbl) in enumerate(zip(st.session_state.points, st.session_state.labels)):
            color = "🟢" if lbl == 1 else "🔴"
            st.write(f"{color} Point {i + 1}: ({pt[0]}, {pt[1]})")

    # Segment button
    if st.button("✂️ RUN SEGMENTATION", type="primary"):
        if not st.session_state.points:
            # Use center point as fallback
            st.session_state.points = [[w // 2, h // 2]]
            st.session_state.labels = [1]
            st.info("ℹ️ No points added — using center of image as foreground.")

        with st.spinner("🧠 Running SAM segmentation... (this may take 5-30 seconds)"):
            start_time = time.time()

            try:
                # Run SAM
                predictor = load_sam_predictor()
                image_np = np.array(st.session_state.image)
                predictor.set_image(image_np)

                masks, scores, _ = predictor.predict(
                    point_coords=np.array(st.session_state.points),
                    point_labels=np.array(st.session_state.labels),
                    multimask_output=True,
                )

                # Select best mask
                best_idx = int(np.argmax(scores))
                st.session_state.mask = masks[best_idx]

                elapsed = time.time() - start_time
                st.success(f"✅ Segmentation completed in {elapsed:.1f} seconds!")

            except Exception as e:
                st.error(f"❌ Error during segmentation: {str(e)}")
                st.session_state.mask = None

    # Display result
    if st.session_state.mask is not None:
        st.subheader("🎨 Results")

        # Overlay
        overlay = create_overlay(st.session_state.image, st.session_state.mask)
        st.image(overlay, caption="Segmentation Overlay", use_column_width=True)

        # Raw mask
        st.image(st.session_state.mask.astype(np.float32), caption="Raw Mask", use_column_width=True)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            mask_pil = Image.fromarray((st.session_state.mask * 255).astype(np.uint8))
            st.download_button(
                "📥 Download Mask",
                mask_pil.tobytes(),
                "sam_mask.png",
                "image/png",
                use_container_width=True
            )
        with col2:
            overlay_pil = Image.fromarray(overlay)
            st.download_button(
                "📥 Download Overlay",
                overlay_pil.tobytes(),
                "sam_overlay.png",
                "image/png",
                use_container_width=True
            )