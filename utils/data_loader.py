import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


def preprocess_medical_image(image, target_size=(256, 256)):
    """Preprocess medical image for U-Net++"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize
    image = image.resize(target_size)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(image).unsqueeze(0)


def preprocess_agricultural_image(image):
    """Preprocess agricultural image for SAM"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return np.array(image)