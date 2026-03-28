import cv2
import numpy as np
import logging

from typing import List
from benchmark.timer_decorator import inference_timer

logger = logging.getLogger(__name__)

class CameraConnection:
    """Context manager for managing camera connection and frame acquisition.

    This class provides a robust way to handle camera resources using OpenCV, 
    ensuring that the camera device is properly initialized and released.

    Attributes:
        camera_index (int): The index of the camera device (default is 0).
        capture (cv2.VideoCapture | None): OpenCV video capture object.
        out (cv2.VideoWriter | None): OpenCV video writer object for saving output.
        input_video_path (str): Path to the input video file (used if camera_index is None).
        output_resolution (tuple): Resolution for the output video file.

    Methods:
        __enter__: Initialize the camera device and set properties.

        read_frame: Capture the current frame from the camera.

        save_frame: Save a frame to the output video file.

        __exit__: Release the camera resources and handle potential errors.

    Example:
        with CameraConnection(camera_index=0) as conn:
            frames = conn.read_frame()
            if frames:
                conn.save_frame(frames[0])
                cv2.imshow('Camera', frames[0])
    """
    def __init__(self, camera_index=None , input_video_path='17431598-hd_1920_1080_60fps.mp4',output_resolution=(1920,1080)):
        self.camera_index = camera_index
        self.capture = None
        self.out = None
        self.input_video_path = input_video_path
        self.output_resolution = output_resolution

    def __enter__(self):
        """Initialize and open the camera connection.

        This method attempts to open the camera and configures the frame 
        dimensions to 640x640.

        Returns:
            self (CameraConnection): The initialized CameraConnection instance.

        Raises:
            RuntimeError: If the camera device cannot be opened.
        """

        self.capture = cv2.VideoCapture(self.camera_index) if self.camera_index is not None else cv2.VideoCapture(self.input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 30 if self.camera_index is not None else 60, self.output_resolution)

        # Ensure the capture device opened successfully before configuring properties
        if not self.capture.isOpened():
            logger.error("Failed to open camera.")
            raise RuntimeError(f"Failed to open camera. Camera index: {self.camera_index}")

        # Configure desired frame size after successful open
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        return self

    @inference_timer(device='cpu')
    def read_frame(self) -> List[np.ndarray | None]:
        """Capture a single frame from the camera.

        This method reads the next video frame and returns it inside a list to
        keep the same interface as a multi-frame reader. If reading fails an
        empty list is returned.

        Returns:
            list[np.ndarray]: A list containing the captured image as a numpy
                array [frame]. Returns an empty list on failure.
        """
        logger.debug("Starting frame reading from camera.")

        # Check if capture is initialized before attempting to read
        if self.capture is None:
            logger.error("Attempted to read frame but capture is not initialized.")
            return []

        
        # Read a frame from the camera
        has_frame, frame = self.capture.read()

        # Check if frame capture failed
        if not has_frame or frame is None:
            if self.camera_index is None:
                pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if pos >= total - 1:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    has_frame, frame = self.capture.read()
                    if not has_frame or frame is None:
                        logger.debug("No frame captured from camera.")
                        return []
            else:
                logger.debug("No frame captured from camera.")
                return []
        
        logger.debug("Frame captured successfully from camera.")
        return [frame]
    
    def save_frame(self, frame: np.ndarray):
        """Save a single frame to the output video file.

        This method writes the provided frame to the video file initialized in
        the __enter__ method. It checks if the video writer is opened before
        attempting to write.

        Args:
            frame (np.ndarray): The image frame to be saved.
        """
        if self.out is not None:
            save_frame = cv2.resize(frame, self.output_resolution)
            self.out.write(save_frame)

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the camera resources.

        This method ensures the camera is released regardless of whether 
        an exception occurred during processing.

        Args:
            exc_type: The type of the exception raised (if any).

            exc_value: The value of the exception raised (if any).

            traceback: The traceback information (if any).
        """
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                logger.exception("Failed to release camera resource")

        if self.out:
            try:
                self.out.release()
            except Exception:
                logger.exception("Failed to release video writer resource")

        if exc_type:
            logger.exception("Exception occurred while connecting to camera: %s %s", exc_type, exc_value)

        self.capture = None
        self.out = None


def draw_detections(img : np.ndarray, boxes: List[List[int]], scores: List[float], class_ids: List[int], class_names: List[str]) -> np.ndarray:
    '''
    Draw bounding boxes and labels on the image.
    Args:
        img (np.ndarray): The original image.

        boxes (List[List[int]]): List of bounding boxes, each box is [x1, y1, x2, y2].

        scores (List[float]): List of confidence scores for each detection.

        class_ids (List[int]): List of class IDs for each detection.

        class_names (List[str]): List of class names corresponding to class IDs.
    Returns:
        np.ndarray: Image with drawn detections.
    '''
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]}: {score:.2f}"

        # Draw bounding box
        color = (0, 255, 0)  # Green color for boxes
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Calculate text size for background box
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Position text above the box if there's space, otherwise below
        text_y1 = y1 - t_h - 5 if y1 - t_h - 5 > 0 else y1 + t_h + 5
        text_y2 = y1 if y1 - t_h - 5 > 0 else y1 + t_h + 15

        # Background box for text
        cv2.rectangle(img, (x1, text_y1), (x1 + t_w, text_y2), color, -1)

        # Put text label
        text_pos_y = y1 - 5 if y1 - t_h - 5 > 0 else y1 + t_h + 10
        cv2.putText(img, label, (x1, text_pos_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img