import cv2
import logging

from pipeline.camera_connection import CameraConnection, draw_detections
from pipeline.preprocess import preprocess
from benchmark.timer_decorator import inference_timer
from config import CLASS_NAMES, SAVE_VIDEO, SHOW_VIDEO
from pipeline.postprocess import PostProcess
from pipeline.inference import PyTorchInferencer

logger = logging.getLogger(__name__)


@inference_timer(device='cpu')
def full_pipeline(conn: CameraConnection, inferencer: PyTorchInferencer, post_processor: PostProcess):
    '''Run the full inference pipeline: capture frame, preprocess, infer, post-process, and display results.
    
    This function is called in a loop to continuously process frames from the camera.

        Args:
        conn (CameraConnection): The camera connection object for reading frames and saving output.
        
        inferencer (PyTorchInferencer): The inference engine for running the model on preprocessed frames.
        
        post_processor (PostProcess): The post-processing object for converting raw model outputs into structured detections.
    '''
    frames = conn.read_frame() # Read a single frame from the camera
    if not frames:
        logger.warning("No frames captured.")
    else:
        im, metadatas = preprocess(frames) # Preprocess the frame for model input
        raw_output = inferencer(im) # Run inference on the preprocessed frame
        logger.debug(f"Output shape: {raw_output.shape}")
        output = post_processor(raw_output, metadatas) # Post-process the raw output to get structured detections
        logger.debug(f"Post-processed results: {output}")
        

        
        if SHOW_VIDEO or SAVE_VIDEO:
            boxes = [d['box'] for d in output[0]]
            scores = [d['score'] for d in output[0]]
            class_ids = [d['class_id'] for d in output[0]]
            output_img = draw_detections(frames[0], 
                                        boxes=boxes, scores=scores, class_ids=class_ids, class_names=CLASS_NAMES)  # Display the frame with detections if benchmarking is disabled
        if SHOW_VIDEO:
            cv2.imshow("Camera Feed", output_img) # Display the frame with detections drawn on it
        if SAVE_VIDEO:
            conn.save_frame(output_img)  # Save the output frame to video

