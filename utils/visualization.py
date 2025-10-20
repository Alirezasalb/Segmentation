import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay mask on image
    Args:
        image: Original image (PIL or numpy)
        mask: Binary mask (numpy array)
        alpha: Transparency of mask
        color: Color of mask overlay (BGR for OpenCV)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color

    # Overlay
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    return overlay


def visualize_results(original, prediction, title="Segmentation Result"):
    """Visualize original image and prediction side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    if isinstance(original, torch.Tensor):
        original = original.squeeze().permute(1, 2, 0).cpu().numpy()
        original = (original * 0.5 + 0.5)  # Denormalize

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Prediction
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig