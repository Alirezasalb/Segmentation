import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2


class SAMWrapper:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_b_01ec64.pth"):
        """
        Initialize SAM model
        Note: You need to download the checkpoint from https://github.com/facebookresearch/segment-anything#model-checkpoints
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image):
        """Set the image for segmentation"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None):
        """
        Predict masks using SAM
        Args:
            point_coords: Nx2 array of point coordinates
            point_labels: N array of point labels (1 for foreground, 0 for background)
            box: 4-element array [x1, y1, x2, y2]
            mask_input: mask input for refinement
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
        )

        # Return the highest scoring mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]