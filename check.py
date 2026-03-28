
from config import CLASS_NAMES
from pipeline.camera_connection import draw_detections
from benchmark.mAP_calculator import MAPCalculator
import cv2
import logging
import os
import numpy as np
from pipeline.camera_connection import CameraConnection, draw_detections
from pipeline.preprocess import preprocess
from benchmark.timer_decorator import inference_timer
from config import CLASS_NAMES
from pipeline.postprocess import PostProcess
from pipeline.inference import PyTorchInferencer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# calc = MAPCalculator()
# postprocesser = PostProcess(conf_threshold=0.001, iou_threshold=0.45)
# calc.post_processor = postprocesser
# calc.predict(batch_size=8)
# mAP50 = calc.calculate_mAP(iou_threshold=0.5)

# print(f"mAP@0.5: {mAP50:.4f}")


# calc = MAPCalculator()
# file_name = '000000000139'
# input_data = {
#     "file_name": file_name,
#     "height": 426,
#     "width": 640
# }

# print(1)
# image, _ = calc._get_images(file_name)
# image = image[0] # Get the single image from the list
# calc.load_all_labels(input_data) # Get ground truth labels from text files
# labels = calc.gt_by_class
# print(labels)
# print(type(image), image.shape)
# boxes = []
# class_ids = []
# for i in range(80):
#     if len(labels[i]) == 0:
#         continue
#     for label in labels[i][file_name]:
#         boxes.append(label['box'])
#         class_ids.append(i)

# length = len(boxes)
# output_img = draw_detections(image, boxes=boxes, scores=[0.5]*length, class_ids=class_ids, class_names=CLASS_NAMES)
# cv2.imshow('Ground Truth', output_img)
# cv2.waitKey(0)

print(0)
calc = MAPCalculator()
labels = calc.gt_by_class
images = []
index = []

print(0.5)

file_names = [d['file_name'] for d in calc.data]
for file_name in file_names:
            img = cv2.imread(os.path.join(calc.image_path, f"{file_name}.jpg"))
            if img is not None:
                images.append(img)
                index.append(file_name)

print(1)
inferencer = PyTorchInferencer(model_path="yolo11n.yaml", weights_path="pure_weight.pt", device="cpu")
for idx, frames in enumerate(images):
    gt_boxes = []
    gt_class_ids = []
    file_name = index[idx]
    for i in range(80):
        if len(labels[i]) == 0:
            continue
        
        if file_name in labels[i]:
            for label in labels[i][file_name]:
                gt_boxes.append(label['box'])
                gt_class_ids.append(i)
    print(frames.shape)
    im, metadatas = preprocess([frames]) # Preprocess the frame for model input
    raw_output = inferencer(im) # Run inference on the preprocessed frame
    logger.debug(f"Output shape: {raw_output.shape}")
    post_processor = PostProcess()
    output = post_processor(raw_output, metadatas) # Post-process the raw output to get structured detections
    print(output)
    logger.debug(f"Post-processed results: {output}")
    boxes = [d['box'] for d in output[0]]
    scores = [d['score'] for d in output[0]]
    class_ids = [d['class_id'] for d in output[0]]
    output_img1 = draw_detections(frames.copy(), 
                                boxes=boxes, scores=scores, class_ids=class_ids, class_names=CLASS_NAMES)
    logger.debug(f"output_img1 shape: {output_img1.shape}")
    output_img2 = draw_detections(frames.copy(), 
                                boxes=gt_boxes, scores=[0.99]*len(gt_boxes), class_ids=gt_class_ids, class_names=CLASS_NAMES)
    logger.debug(f"output_img2 shape: {output_img2.shape}")

    cv2.imshow("Camera Feed", output_img1)
    cv2.imshow("Ground Truth", output_img2)
    cv2.waitKey(0)


