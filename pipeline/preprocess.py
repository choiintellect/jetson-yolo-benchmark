import numpy as np
import torch
import cv2
import logging
from typing import List, Dict, Tuple
from benchmark.timer_decorator import inference_timer
from config import TARGET_DEVICE

logger = logging.getLogger(__name__)

@inference_timer(device='cpu')
def preprocess(frames: List[np.ndarray]) -> tuple[torch.Tensor, List[Dict]]:
    """Prepare input image before inference.

    Args:
        frames (List[np.ndarray]): List of images, each with shape (H, W, 3).

    Returns:
        im (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W) with dtype 
        float32 in [0,1].

        metadatas (List[Dict]): List of metadata dictionaries for each image.
            Each dictionary contains:
            - "ratio": The scaling ratio applied to the original image.
            - "pad": A tuple (pad_left, pad_top) indicating the number of pixels padded on the left and top sides.
            - "orig_shape": The original shape of the input image as a tuple (height, width).
    """
    logger.debug(f"Starting preprocessing.")

    transformed_images = []
    metadatas = []

    # Initialize the LetterBox transformer with the desired output shape and settings
    letterbox = LetterBox(new_shape=(640, 640), auto=False)

    # Apply letterbox transformation to each frame and collect metadata
    for x in frames:
        img, meta = letterbox(x)
        transformed_images.append(img)
        metadatas.append(meta)

    device = torch.device(TARGET_DEVICE)

    im = np.stack(transformed_images, axis=0)
    im = im[..., ::-1]  # BGR to RGB
    im = im.transpose(0, 3, 1, 2)  # (N, H, W, 3) to (N, 3, H, W)

    # Normalize pixel values to [0, 1] and convert to float32
    im = np.ascontiguousarray(im.astype(np.float32) / 255.0)
    im = torch.from_numpy(im).to(device=device)

    logger.debug(f"Preprocessed image tensor shape: {im.shape}, dtype: {im.dtype}")
    return im, metadatas

class LetterBox:
    """Resize image and padding for detection.

    This class resizes and pads images to a specified shape while preserving aspect ratio.

    Attributes:
        new_shape (tuple[int, int]): Target shape (height, width) for resizing.

        auto (bool): Whether to use minimum rectangle.

        scale_fill (bool): Whether to stretch the image to new_shape.

        scaleup (bool): Whether to allow scaling up. If False, only scale down.

        stride (int): Stride for rounding padding.

        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image according to the specified parameters.

    Example:
        letterbox = LetterBox(new_shape=(640, 640), auto=False)
        transformed_image, metadata = letterbox(image)
    """
    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32, center=True):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, image : np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
        """Resize and pad an image for object detection.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its aspect ratio and adding padding to fit the new shape.

        Args:
            image (np.ndarray):  The input image with shape (H, W, 3) as a numpy array.

        Returns:
            image (np.ndarray): returns the resized and padded image with shape (H, W, 3).

            metadata (Dict[str, any]): A dictionary containing the following metadata about the   transformation (for post-processing):
                - "ratio": The scaling ratio applied to the original image.
                - "pad_left": The number of pixels padded on the left side.
                - "pad_top": The number of pixels padded on the top side.
                - "orig_shape": The original shape of the input image as a tuple (height, width).
        """
        
        
        shape = image.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape  # target shape [height, width]

        # Scale ratio (uniform) by default
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        if not self.scaleup:  # only scale down, do not scale up
            r = min(r, 1.0)

        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        # If auto is True, adjust padding to be a multiple of stride (for efficient inference on some hardware)
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        # If scale_fill is True, stretch the image to fill the new shape, ignoring aspect ratio
        elif self.scale_fill:
            # When scale_fill is used we stretch independently in width/height.
            # Compute non-uniform ratios for x and y so postprocessing can invert correctly.
            r_w = new_shape[1] / shape[1]
            r_h = new_shape[0] / shape[0]
            if not self.scaleup:
                r_w = min(r_w, 1.0)
                r_h = min(r_h, 1.0)
            # set new_unpad to target shape (width, height)
            new_unpad = new_shape
            dw, dh = 0.0, 0.0
            # store non-uniform ratio as a tuple in metadata
            r = (r_w, r_h)

        # If center is True, divide padding into 2 sides
        if self.center:
            left = int(np.floor(dw / 2.0))
            right = int(np.ceil(dw / 2.0))
            top = int(np.floor(dh / 2.0))
            bottom = int(np.ceil(dh / 2.0))
        else:
            left, top, right, bottom = 0, 0, int(dw), int(dh)

        # Resize the image
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Update metadata for post-processing. If scale_fill was used, 'ratio' may be a
        # tuple (ratio_w, ratio_h); otherwise it's a single uniform float.
        metadata = {
            "ratio": r,
            "pad": (left, top),
            "orig_shape": shape
        }

        logger.debug(f"LetterBox transformation applied: ratio={r}, pad_left={left}, pad_top={top}, orig_shape={shape}, new_shape={image.shape[:2]}")

        return image, metadata