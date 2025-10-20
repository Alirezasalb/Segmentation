import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import tempfile

# Import our modules
from models.unet_plusplus import UNetPlusPlus
from models.sam_wrapper import SAMWrapper
from utils.data_loader import preprocess_medical_image, preprocess_agricultural_image
from utils.visualization import overlay_mask, visualize_results

# Set page config
st.set_page_config(
    page_title="Advanced Segmentation Tool",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Advanced Segmentation for Medical & Agricultural Applications")
st.markdown("""
This application demonstrates advanced segmentation techniques:
- **U-Net++** for medical imaging (MRI tumor segmentation)
- **SAM (Segment Anything Model)** for agricultural applications (leaf disease detection)
""")

# Sidebar for application selection
st.sidebar.header("Application Settings")
app_mode = st.sidebar.selectbox(
    "Choose Application",
    ["Medical Imaging (MRI Tumor Segmentation)", "Agricultural Imaging (Leaf Disease Detection)"]
)

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

# Medical Imaging Section
if "Medical" in app_mode:
    st.header("🧠 Medical Imaging: MRI Tumor Segmentation")


    # Load pre-trained model
    @st.cache_resource
    def load_medical_model():
        model = UNetPlusPlus(num_classes=1, input_channels=3)
        # In a real application, you would load weights from a trained model
        # For demo purposes, we'll use a randomly initialized model
        model.eval()
        return model.to(device)


    model = load_medical_model()

    # Upload medical image
    uploaded_file = st.file_uploader("Upload an MRI scan", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        # Preprocess image
        input_tensor = preprocess_medical_image(image)
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.sigmoid(prediction)

        # Convert to numpy for visualization
        pred_mask = prediction.squeeze().cpu().numpy()

        # Create overlay
        overlay = overlay_mask(image, pred_mask, alpha=0.6, color=(255, 0, 0))

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original MRI", use_column_width=True)

        with col2:
            st.image(pred_mask, caption="Predicted Tumor Mask", use_column_width=True)

        with col3:
            st.image(overlay, caption="Overlay", use_column_width=True)

        # Download options
        st.subheader("Download Results")
        mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
        overlay_pil = Image.fromarray(overlay)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Mask",
                data=mask_pil.tobytes(),
                file_name="tumor_mask.png",
                mime="image/png"
            )
        with col2:
            st.download_button(
                label="Download Overlay",
                data=overlay_pil.tobytes(),
                file_name="tumor_overlay.png",
                mime="image/png"
            )

    else:
        st.info("Please upload an MRI scan to perform tumor segmentation.")

        # Show example
        st.subheader("Example")
        example_path = "data/medical/example_mri.jpg"
        if os.path.exists(example_path):
            example_img = Image.open(example_path)
            st.image(example_img, caption="Example MRI Scan", use_column_width=True)

# Agricultural Imaging Section
else:
    st.header("🌿 Agricultural Imaging: Leaf Disease Detection")


    # Load SAM model
    @st.cache_resource
    def load_sam_model():
        # Note: In a real deployment, you would need to download the SAM checkpoint
        # For this demo, we'll simulate the functionality
        try:
            # This would normally load the actual model
            # sam = SAMWrapper(checkpoint_path="sam_vit_h_4b8939.pth")
            # return sam
            return "SAM Model (simulated)"
        except Exception as e:
            st.warning(f"Could not load SAM model: {e}")
            return None


    sam_model = load_sam_model()

    # Upload agricultural image
    uploaded_file = st.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        # Preprocess image
        processed_image = preprocess_agricultural_image(image)

        # Simulate SAM segmentation
        # In a real implementation, you would use:
        # sam_model.set_image(processed_image)
        # mask, score = sam_model.predict(...)

        # For demo purposes, we'll create a simulated mask
        h, w = processed_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create a simulated diseased region (for demo)
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        y, x = np.ogrid[:h, :w]
        mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[mask_area] = 1

        # Create overlay
        overlay = overlay_mask(processed_image, mask, alpha=0.6, color=(0, 0, 255))  # Blue for disease

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original Leaf", use_column_width=True)

        with col2:
            st.image(mask, caption="Disease Mask", use_column_width=True)

        with col3:
            st.image(overlay, caption="Disease Overlay", use_column_width=True)

        # Disease analysis
        disease_area = np.sum(mask) / mask.size * 100
        st.subheader("Disease Analysis")
        st.metric("Affected Area", f"{disease_area:.2f}%")

        if disease_area > 10:
            st.warning("⚠️ Significant disease detected! Consider treatment.")
        elif disease_area > 2:
            st.info("ℹ️ Minor disease detected. Monitor closely.")
        else:
            st.success("✅ Healthy leaf detected!")

        # Download options
        st.subheader("Download Results")
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        overlay_pil = Image.fromarray(overlay)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Disease Mask",
                data=mask_pil.tobytes(),
                file_name="disease_mask.png",
                mime="image/png"
            )
        with col2:
            st.download_button(
                label="Download Overlay",
                data=overlay_pil.tobytes(),
                file_name="disease_overlay.png",
                mime="image/png"
            )

    else:
        st.info("Please upload a leaf image to detect diseases.")

        # Show example
        st.subheader("Example")
        example_path = "data/agricultural/example_leaf.jpg"
        if os.path.exists(example_path):
            example_img = Image.open(example_path)
            st.image(example_img, caption="Example Leaf Image", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Note**: 
- For medical applications, this demo uses a randomly initialized U-Net++ model. In practice, you would need to train the model on medical datasets.
- For agricultural applications, SAM requires a checkpoint file which is not included due to size constraints. The demo simulates the functionality.
- Always consult with medical professionals for actual diagnosis.
""")