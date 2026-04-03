import json
import cv2
import numpy as np
import os
import logging

from typing import Optional, List, Dict
from pipeline.postprocess import PostProcess
from pipeline.preprocess import preprocess
from pipeline.inference import PyTorchInferencer

from collections import defaultdict

logger = logging.getLogger(__name__)

class MAPCalculator:
    '''Class to calculate mean Average Precision (mAP) for object detection.
    This class loads ground truth labels, runs inference to get predictions, and calculates mAP based on IoU matching.

    Attributes:
        json_path (str): Path to the JSON file containing image metadata.

        label_path (str): Path to the directory containing ground truth label files. (.txt file)

        image_path (str): Path to the directory containing image files. (.jpg file)

        num_classes (int): The number of object classes in the dataset.

        data (List[Dict[str, any]]): A list of dictionaries containing image metadata loaded from the JSON file.

        inferencer (Optional[PyTorchInferencer]): The inference engine to use for generating predictions.

        post_processor (Optional[PostProcess]): The post-processing pipeline to use for refining predictions.

        gt_by_class (Dict[int, Dict[str, List[Dict[str, float]]]): A dictionary mapping class IDs to dictionaries that map image IDs to lists of ground truth boxes.

        pred_by_class (Dict[int, List[Dict[str, float]]]): A dictionary mapping class IDs to lists of predicted detections.

        tp_by_class (Dict[int, List[bool]]): A dictionary mapping class IDs to lists of true positive flags for predictions.

        AP_by_class (Dict[int, float]): A dictionary mapping class IDs to their calculated Average Precision (AP) values.

        mAPs (Dict[str, float]): A dictionary to store calculated mAP values for different IoU thresholds or evaluation settings.

    Methods:
        calculate_mAP: Calculate mean Average Precision (mAP) for object detection.

        _calculate_AP: Calculate Average Precision (AP) for a specific class.

        _match_predictions_to_ground_truths: Match predicted boxes to ground truth boxes based on IoU and update true positive flags.

        predict: Run inference on the dataset and populate predictions for mAP calculation.
        
        load_all_labels: Load ground truth labels from text files and organize them by class and image ID.

        _get_images: Load images from the specified path based on file names.

        _inference: Run inference on a batch of images using the specified inferencer and post-processor.

        _parse_json: Parse the JSON file to extract image metadata.

        reset_for_new_mAP: Reset internal state for a new mAP calculation.

        reset_for_new_inference: Reset internal state for a new inference run, including predictions and mAP-related data.
    
    Example:
        calc = MAPCalculator()
        calc.predict(batch_size=8)
        mAP50 = calc.calculate_mAP(iou_threshold=0.5)

        calc.reset_for_new_mAP()
        mAP75 = calc.calculate_mAP(iou_threshold=0.75)

        calc.reset_for_new_inference()
        calc.inferencer = AnotherInferencer()
        calc.post_processor = AnotherPostProcessor()
        calc.predict(batch_size=8)
        mAP50_new = calc.calculate_mAP(iou_threshold=0.5)

    '''

    def __init__(self, json_path : str ='./data/instances_val2017.json',label_path : str ='./data/labels/val2017', image_path : str ='./data/images/val2017', inferencer : Optional[PyTorchInferencer] = None, post_processor : Optional[PostProcess] = None, num_classes : int = 80):
        self.json_path = json_path
        self.label_path = label_path
        self.image_path = image_path
        self.num_classes = num_classes
        self.data = self._parse_json()
        self.inferencer = inferencer
        self.post_processor = post_processor
        self.gt_by_class = None
        self.pred_by_class = {c: [] for c in range(num_classes)}
        self.tp_by_class = {c: [] for c in range(num_classes)}
        self.AP_by_class = {c: 0.0 for c in range(num_classes)}
        self.mAPs = {}

        logger.debug(f"Initialized MAPCalculator with JSON path: {json_path}, label path: {label_path}, image path: {image_path}, number of classes: {num_classes}")
        logger.debug(f"Loaded {len(self.data)} entries from JSON file.")
        self.load_all_labels(self.data)

    def calculate_mAP(self, iou_threshold: float = 0.5) -> float:
        '''Calculate mean Average Precision (mAP) for object detection.

        Args:
            iou_threshold (float): IoU threshold to consider a detection as true positive.

        Returns:
            float: The calculated mAP value.
        '''
        logger.debug(f"Starting mAP calculation with IoU threshold: {iou_threshold}")  
        # Reset internal state for a new mAP calculation.
        self.reset_for_new_mAP()

        # Match predictions to ground truths and update true positive flags
        self._match_predictions_to_ground_truths(iou_threshold)

        # Calculate AP for each class and store in AP_by_class
        for class_id in range(self.num_classes):
            num_gt = sum([len(v) for v in self.gt_by_class[class_id].values()])

            # If there are ground truths and predictions, calculate AP. 
            # If there are no ground truths, set AP to NaN to indicate it's not applicable. # If there are ground truths but no predictions, AP will be 0.0.
            if num_gt > 0:
                self.AP_by_class[class_id] = self._calculate_AP(class_id) if len(self.pred_by_class[class_id]) > 0 else 0.0
            else:
                self.AP_by_class[class_id] = float('nan') 

        # Calculate mAP, ignoring NaN values for classes with no ground truths
        self.mAPs[iou_threshold] = np.nanmean(list(self.AP_by_class.values()))
        logger.debug(f"Finished mAP calculation. mAP@{iou_threshold:.2f}: {self.mAPs[iou_threshold]:.4f}")
        return self.mAPs[iou_threshold]

    def _calculate_AP(self, class_id: int) -> float:
        '''Calculate Average Precision (AP) for a specific class.

        Args:
            class_id (int): The class ID for which to calculate AP.

        Returns:
            float: The calculated AP value for the specified class.
        '''
        logger.debug(f"Starting AP calculation for class: {class_id}")
        # Get true positive flags and false positive flags for the specified class
        tp = np.array(self.tp_by_class[class_id])
        fp = 1 - tp

        # Calculate cumulative sums of true positives and false positives
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        # Calculate precision and recall
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall = tp_cum / (sum([len(v) for v in self.gt_by_class[class_id].values()]) + 1e-6)

        # Interpolate precision at standard recall levels (0.0 to 1.0 with step of 0.01)
        recall_levels = np.linspace(0,1,101)
        precisions = []

        # For each recall level, find the maximum precision where recall is greater than or equal to that level
        for r in recall_levels:
            p = precision[recall >= r]
            precisions.append(np.max(p) if len(p) else 0)

        # Calculate AP as the mean of the interpolated precisions at the standard recall levels
        ap = np.mean(precisions)

        logger.debug(f"Finished AP calculation for class: {class_id}. AP: {ap:.4f}")
        return ap
    
    def _match_predictions_to_ground_truths(self, iou_threshold: float = 0.5) -> None:
        '''Match predicted boxes to ground truth boxes based on IoU and update true positive flags.

        Args:
            iou_threshold (float): IoU threshold to consider a detection as true positive.
        '''
        logger.debug(f"Matching predictions to ground truths with IoU threshold: {iou_threshold}")
        # Match predicted boxes to ground truth boxes based on IoU
        for class_id in range(self.num_classes):
            preds = self.pred_by_class[class_id]
            gts_for_class = self.gt_by_class[class_id] 

            for pred in preds:
                image_id = pred['image_id']
                pred_box = pred['box']
                
                best_iou = -1
                best_gt = None

                # Check if there are ground truths for this image and class. If not, this prediction is a false positive.
                if image_id in gts_for_class:
                    for gt in gts_for_class[image_id]:
                        iou = calculate_iou(pred_box, gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt
                
                # A prediction is a true positive if it has the highest IoU with a ground truth box that exceeds the threshold and has not been matched yet. Otherwise, it's a false positive.
                if best_iou >= iou_threshold and best_gt is not None and not best_gt['matched']:
                    self.tp_by_class[class_id].append(1)
                    best_gt['matched'] = True
                else:
                    self.tp_by_class[class_id].append(0)
        logger.debug(f"Finished matching predictions to ground truths. Updated true positive flags for each class.")
        return

    def predict(self, batch_size : int = 8) -> None:
        '''
        Run inference on the dataset and populate predictions for mAP calculation.

        Args:
            batch_size (int): The number of images to process in each batch during inference.

        '''
        logger.info(f"Starting prediction with batch size: {batch_size}")

        logger.info(f"Total number of images to process: {len(self.data)}")
        for i in range(0, len(self.data), batch_size):
            logger.info(f"Processing batch {i // batch_size + 1}")
            batch_data = self.data[i:i+batch_size]

            batch_ids = [d['file_name'] for d in batch_data]

            # Get images for the current batch and predictions for those images
            
            images, index = self._get_images(batch_ids)
            preds_for_images = self._inference(images) # type(preds_for_image) == List[Dict(image_id=str, class_id=int, box=List[float], score=float)]
            # Organize predictions by class and image ID
            for idx, preds_for_image in enumerate(preds_for_images): 
                for pred in preds_for_image:
                    pd = {
                        "image_id": index[idx],
                        "box": pred['box'],
                        "score": pred['score']
                    }
                    self.pred_by_class[pred['class_id']].append(pd)
        
        # Sort predictions for each class by confidence score in descending order.
        self.pred_by_class = {c: sorted(preds, key=lambda x: x['score'], reverse=True) for c, preds in self.pred_by_class.items()}

        logger.debug(f"Finished prediction and sorting predictions by confidence score for each class.")
        return

    def load_all_labels(self, data: List[Dict[str, any]]) -> None:
        '''Load ground truth labels from text files and organize them by class and image ID.

        Args:
            data (List[Dict[str, any]]): A list of dictionaries containing image metadata, each with keys:
                - "file_name": The name of the image file (without extension).
                - "height": The height of the image in pixels.
                - "width": The width of the image in pixels.
        '''
        logger.debug(f"Starting to load ground truth labels from text files in path: {self.label_path}")
        gt_by_class = {
            c: defaultdict(list) for c in range(self.num_classes)
        }

        if not isinstance(data, list):
            data = [data]

        for item in data:

            file_name = item["file_name"]
            height = item["height"]
            width = item["width"]
            
            # Open the corresponding label file for the image and read the ground truth boxes. Each line in the label file is expected to be in the format:
            #  class_id(int) cx(float) cy(float) w(float) h(float) (normalized coordinates).
            file_path = os.path.join(self.label_path, f"{file_name}.txt")

            if not os.path.exists(file_path):
                continue
            
            # Convert the normalized coordinates to absolute pixel values and store them in gt_by_class
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:

                    line = line.strip()
                    if not line:
                        continue

                    class_id, cx, cy, w, h = line.split()

                    class_id = int(class_id)

                    cx = float(cx) * width
                    cy = float(cy) * height
                    w = float(w) * width
                    h = float(h) * height

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    gt = {
                        "box": [x1, y1, x2, y2],
                        "matched": False
                    }
                    gt_by_class[class_id][file_name].append(gt)
        self.gt_by_class = gt_by_class
        logger.debug(f"Finished loading ground truth labels. Organized labels for {len(gt_by_class)} classes.")
        return

    def _get_images(self, file_names : List[str]) -> tuple[List[np.ndarray], List[str]]:
        '''Get images from the specified path. Only .jpg files are supported.

        Args:
            file_names (List[str]): List of file names (without extension) to load.

        Returns:
            tuple[List[np.ndarray], List[str]]: A tuple containing a list of loaded images as numpy arrays and their corresponding file names.
        '''
        logger.debug(f"Starting to load images from path: {self.image_path}")
        if not isinstance(file_names, list):
            file_names = [file_names]
        images = []
        index = []
        for file_name in file_names:
            img = cv2.imread(os.path.join(self.image_path, f"{file_name}.jpg"))
            if img is not None:
                images.append(img)
                index.append(file_name)

        logger.debug(f"Finished loading images. Loaded {len(images)} images.")
        return images, index

    def _inference(self, images: List[np.ndarray]) -> List[dict]:
        '''Run inference on a batch of images using the specified inferencer and post-processor.

        Args:
            images (List[np.ndarray]): A list of images as numpy arrays to run inference on.

        Returns:
            List[dict]: A list of dictionaries containing predictions for each image, where each dictionary has keys:
                - "image_id": The identifier of the image (e.g., file name).
                - "predictions": A list of predicted bounding boxes and class IDs.
        '''
        logger.debug(f"Starting inference on batch of {len(images)} images.")
        if self.inferencer is None:
            self.inferencer = PyTorchInferencer()
        if self.post_processor is None:
            self.post_processor = PostProcess()
        im, metadatas = preprocess(images) # Preprocess the frame for model input
        raw_output = self.inferencer(im) # Run inference on the preprocessed frame
        output = self.post_processor(raw_output, metadatas)
        logger.debug(f"Finished inference and post-processing. Generated predictions for {len(output)} images.")
        return output

    def _parse_json(self,) -> List[Dict[str, any]]:
        '''Parse the JSON file to extract image metadata.

        Returns:
            List[Dict[str, any]]: A list of dictionaries containing image metadata, each with keys:
                - "file_name": The name of the image file (without extension).
                - "height": The height of the image.
                - "width": The width of the image.
        ''' 

        logger.debug(f"Starting parsing JSON file at path: {self.json_path}")
        data = []

        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            images = json_data ['images']
            for img in images:
                h = img.get('height')
                w = img.get('width')
                fn = img.get('file_name').split('.')[0]

                # Validate that height and width are positive numbers and file name is a non-empty string before adding to the data list.
                if (isinstance(h, (int, float)) and h > 0) and \
                (isinstance(w, (int, float)) and w > 0) and \
                (isinstance(fn, str) and len(fn) > 0):
                    data.append({
                        "file_name": fn,
                        "height": h,
                        "width": w
                    })

        logger.debug(f"Finished parsing JSON file. Extracted metadata for {len(data)} images.")
        return data

    def reset_for_new_mAP(self):
        '''Reset internal state for a new mAP calculation.
        
        This method clears the true positive flags and resets the AP values for each class, but retains the ground truth and predictions for potential reuse.
        '''
        self.tp_by_class = {c: [] for c in range(self.num_classes)}
        self.AP_by_class = {c: 0.0 for c in range(self.num_classes)}
        for classes in self.gt_by_class.values():
            for gts in classes.values():
                for gt in gts:
                    gt['matched'] = False
        
        logger.debug("Reset internal state for new mAP calculation, including true positive flags and AP values for each class, but retaining ground truth and predictions.")

    def reset_for_new_inference(self):
        '''Reset internal state for a new inference run, including predictions and mAP-related data.
        '''
        
        self.pred_by_class = {c: [] for c in range(self.num_classes)}
        self.reset_for_new_mAP()

        logger.debug("Reset internal state for new inference run, including predictions and mAP-related data.")
    
    def print_mAPs(self):
        '''Print the calculated mAP values for different IoU thresholds or evaluation settings.
        '''
        logger.info("Calculated mAP values:")
        for iou_threshold, mAP in self.mAPs.items():
            logger.info(f"mAP@{iou_threshold:0.2f}: {mAP:.4f}")


def calculate_iou(box1: List, box2: List) -> float:
    """Calculate Intersection over Union (IoU) of two boxes in xyxy format.

    Args:
        box1: List of [x1, y1, x2, y2] for the first box.

        box2: List of [x1, y1, x2, y2] for the second box.

    Returns:
        IoU value as a float between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou_value = inter_area / (union_area + 1e-7)
    return iou_value