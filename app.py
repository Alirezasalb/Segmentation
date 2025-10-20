import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

# ----------------------------
# Configuration
# ----------------------------
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_resource
def load_sam_model():
    """Load SAM model once and cache it"""
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # Dodger blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def overlay_mask_on_image(image, mask, alpha=0.6):
    """Overlay binary mask on image"""
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay = image.copy()
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Create colored mask (blue)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 2] = mask_binary  # Blue channel

    # Blend
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

    return overlay


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(
    page_title="SAM Universal Segmenter",
    page_icon="✂️",
    layout="wide"
)

st.title("✂️ SAM Universal Image Segmenter")
st.markdown("""
Upload **any image**, click on **points** to indicate what to **keep (🟢)** or **exclude (🔴)**, then click **Segment**!
- **Green star** = object you want to segment
- **Red star** = background or things to ignore
""")

# Initialize session state
if 'points' not in st.session_state:
    st.session_state.points = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'image' not in st.session_state:
    st.session_state.image = None
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False

# Sidebar
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. Upload an image
    2. Click **'Add Foreground Point'** and click on the image where the object is
    3. (Optional) Click **'Add Background Point'** to exclude areas
    4. Click **'Segment'**
    5. Download the mask or overlay
    """)

    uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.image = image
        # Reset points when new image is uploaded
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.segmentation_done = False

# Main area
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Controls")

    if st.session_state.image is not None:
        if st.button("🟢 Add Foreground Point (Click on object)"):
            st.session_state.point_type = 1
        if st.button("🔴 Add Background Point (Click on background)"):
            st.session_state.point_type = 0

        if st.button("✂️ Segment!"):
            if len(st.session_state.points) == 0:
                st.warning("Please add at least one point!")
            else:
                try:
                    predictor = load_sam_model()
                    image_np = np.array(st.session_state.image)
                    predictor.set_image(image_np)

                    input_points = np.array(st.session_state.points)
                    input_labels = np.array(st.session_state.labels)

                    masks, scores, logits = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True,
                    )

                    # Pick best mask
                    best_idx = np.argmax(scores)
                    st.session_state.mask = masks[best_idx]
                    st.session_state.overlay = overlay_mask_on_image(st.session_state.image, st.session_state.mask)
                    st.session_state.segmentation_done = True
                    st.success("✅ Segmentation complete!")
                except Exception as e:
                    st.error(f"Error during segmentation: {str(e)}")

        if st.button("🔄 Reset Points"):
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.segmentation_done = False
            st.rerun()

with col1:
    if st.session_state.image is not None:
        # Display image and handle clicks
        image_width = 600
        st_image = st.image(st.session_state.image, width=image_width, caption="Click on the image to add points")

        # Use st.canvas or coordinate input? → We'll use a workaround with number inputs
        st.markdown("###