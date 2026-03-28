import torch
import numpy as np
import logging
from typing import List, Dict
from benchmark.timer_decorator import inference_timer

logger = logging.getLogger(__name__)

class PostProcess:
    """Post-processing for raw model output.

    Converts raw model outputs to a list of detections per image, applies
    class-wise NMS, scales boxes back to original image coordinates, and
    returns a list of detection dicts per image (List[List[Dict]]).

    Detection dict format:
        {"image_id": int, "class_id": int, "score": float, "box": [x1,y1,x2,y2]}

    Attributes:
        conf_threshold (float): Confidence threshold for filtering detections.

        iou_threshold (float): IoU threshold for NMS.
    
    Methods:
    __call__: Process raw model output and return structured detections.

    Example:
        post_processor = PostProcess(conf_threshold=0.25, iou_threshold=0.45)
        detections = post_processor(raw_output, metas)
    """

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

    @inference_timer(device='cpu')
    def __call__(self, raw_output: torch.Tensor, metas: List[dict]) -> List[List[Dict]]:
        """Process raw model output into per-image detection lists.

        Args:
            raw_output: torch.Tensor, expected shape (N, 84, 8400).

            metas: list of metadata dicts (one per image) with keys:
                   - "ratio": scaling ratio
                   - "pad": (pad_left, pad_top)
                   - "orig_shape": (orig_h, orig_w)

        Returns:
            List of per-image lists of detection dicts.
                Each detection dict has keys:
                - "image_id": int (index of the image in the batch)
                - "class_id": int
                - "score": float
                - "box": [x1, y1, x2, y2] in original image coordinates
        """
        logger.debug("Starting postprocessing.")

        # Validate input types
        if not isinstance(raw_output, torch.Tensor):
            raise TypeError("raw_output must be a torch.Tensor")

        if not isinstance(metas, List):
            raise TypeError("metas must be a list of dicts")

        # Convert raw output to numpy for processing
        out = raw_output.detach().cpu().numpy()
        batch_size = out.shape[0]
        results_per_image: List[List[Dict]] = []

        for i in range(batch_size):
            target = out[i]
            # Normalize shape to (num_boxes, data)
            if target.shape[0] < target.shape[1]:
                target = target.T

            # If no boxes, skip to next image
            if target.size == 0:
                results_per_image.append([])
                continue

            # Extract box coordinates and class logits
            boxes_cxcywh = target[:, :4] # In pixel coords, format cx,cy,w,h
            class_logits = target[:, 4:] # Shape (num_boxes, num_classes)
            class_ids = np.argmax(class_logits, axis=1) # Shape (num_boxes,)
            class_scores = np.max(class_logits, axis=1) # Shape (num_boxes,)

            # Filter by confidence threshold
            keep_mask = class_scores >= self.conf_threshold
            if not np.any(keep_mask):
                results_per_image.append([])
                continue

            boxes = boxes_cxcywh[keep_mask] # Pixel coords in cx,cy,w,h format

            
            scores = class_scores[keep_mask] # Confidence scores for the kept boxes
            classes = class_ids[keep_mask] # Class IDs for the kept boxes

            # Convert cx,cy,w,h -> x1,y1,x2,y2
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            xyxy = np.stack([x1, y1, x2, y2], axis=1)

            detections: List[Dict] = []

            # Perform class-wise NMS
            unique_classes = np.unique(classes) # Remove duplicates to get unique class IDs
            for cls in unique_classes:
                idxs = np.where(classes == cls)[0]
                cls_boxes = xyxy[idxs]
                cls_scores = scores[idxs]
                
                # Apply NMS to the class-specific boxes
                keep = nms_numpy(cls_boxes, cls_scores, self.iou_threshold)
                for k in keep:
                    box = cls_boxes[k].copy()
                    score = float(cls_scores[k])
                    detections.append({
                        "image_id": i,
                        "class_id": int(cls),
                        "score": score,
                        "box": box,
                    })

            # Scale boxes back to original image coords and finalize dicts
            if detections:
                boxes_np = np.stack([d["box"] for d in detections], axis=0)
                meta = metas[i]
                pad_left, pad_top = meta["pad"]
                boxes_np = scale_boxes(boxes_np, meta["ratio"], pad_left, pad_top, meta["orig_shape"])
                for idx, d in enumerate(detections):
                    d["box"] = boxes_np[idx].tolist()

            results_per_image.append(detections)

        logger.debug(f"Postprocessing finished.")
        return results_per_image


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Perform standard NMS and return indices of boxes to keep.
    
    Args:
        boxes: numpy array of shape (N, 4) in xyxy format.

        scores: numpy array of shape (N,) with confidence scores.

        iou_threshold: float, IoU threshold for suppression.

    Returns:
        List of indices of boxes to keep after NMS.
    """
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0] 
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas of the boxes
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] # Sort scores in descending order and get indices

    keep: List[int] = []

    # Standard NMS loop
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        ious = inter / (union + 1e-7)

        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    logger.debug(f"NMS finished. Kept {len(keep)} boxes after suppression.")
    return keep

def scale_boxes(boxes: np.ndarray, ratio, pad_left: float, pad_top: float, img_shape) -> np.ndarray:
    """Scale and clip boxes back to original image coordinates.

    Supports both uniform ratio (float) and non-uniform ratio (tuple/list of two floats: (ratio_w, ratio_h)).

    Args:
        boxes: numpy array of shape (N, 4) in xyxy format (scaled coordinates).

        ratio: scaling ratio used during preprocessing. Either a single float (uniform
               scaling) or a tuple (ratio_w, ratio_h) for non-uniform stretch.

        pad_left: horizontal padding added during preprocessing.

        pad_top: vertical padding added during preprocessing.

        img_shape: tuple (orig_h, orig_w) of the original image dimensions.
    
    Returns:
        numpy array of shape (N, 4) with boxes scaled back to original image coordinates and clipped to image boundaries.
    """
    if boxes.size == 0:
        return boxes

    boxes = boxes.astype(np.float32)

    # Support non-uniform ratios when image was stretched (scale_fill=True)
    if isinstance(ratio, (list, tuple, np.ndarray)):
        ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / ratio_w
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / ratio_h
    else:
        r = float(ratio)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / r

    # Clip to image size (width, height)
    orig_h, orig_w = img_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    logger.debug(f"Scaled boxes back to original image coordinates.")
    return boxes