import cv2
import logging
import numpy as np
import argparse
import os

import config

logging.basicConfig(format='%(message)s', level=logging.INFO)
parser = argparse.ArgumentParser(description='Jetson YOLOv11 Benchmarking Tool')

parser.add_argument('--device', type=str, default=config.TARGET_DEVICE, help='cpu or cuda')
parser.add_argument('--show', action='store_true', help='Show video window')
parser.add_argument('--nosave', action='store_true', help='Disable video saving')

parser.add_argument('--mode', type=str, default='run', 
                    choices=['run', 'latency', 'hardware', 'map'])
parser.add_argument('--frames', type=int, default=1000, help='Number of frames to process')
parser.add_argument('--camera', type=int, default=None, help='Use camera input instead of video file. Provide camera index (e.g., 0 for default webcam)')
args = parser.parse_args()

config.TARGET_DEVICE = args.device
if args.mode == 'latency':
    config.TIMER = True
config.SHOW_VIDEO = args.show
if args.nosave:
    config.SAVE_VIDEO = False

from pipeline.camera_connection import CameraConnection
from pipeline.preprocess import preprocess
from benchmark.timer_decorator import inference_timer
from pipeline.postprocess import PostProcess
from pipeline.inference import PyTorchInferencer
from pipeline.full_pipeline import full_pipeline
from benchmark.mAP_calculator import MAPCalculator
from benchmark.hardware_monitor import HardwareMonitor, get_ram


logger = logging.getLogger(__name__)

def just_run(conn: CameraConnection, inferencer: PyTorchInferencer, post_processor: PostProcess, frames_to_process: int = 1000):
    '''
    Run the full inference pipeline in a loop to capture frames from the camera, process them, and display/save the output video.

    SAVE_VIDEO = True recommended to save video.
    SHOW_VIDEO = True if you want to show video.
    '''
    attempts = 0
    while cv2.waitKey(delay = 1) < 0:

        full_pipeline(conn, inferencer, post_processor)

        attempts += 1
        if attempts % 100 == 0:
            logger.info(f"Processed {attempts} frames...")
        if attempts >= frames_to_process:
            break
    cv2.destroyAllWindows()



def bench_latencies(conn: CameraConnection, inferencer: PyTorchInferencer, post_processor: PostProcess, frames_to_process: int = 1000):
    '''
    Run the full inference pipeline in a loop to benchmark latencies of each step (preprocess, inference, post-process) and the full pipeline.
    This function collects latency data and logs average, 99th percentile, and jitter for each step.

    TIMER = True required to enable latency benchmarking.
    '''
    logger.info("Starting warm-up phase...")

        # Warm up the model with 100 frames before starting timed inference
        # Reuse a single PostProcess instance to avoid allocation overhead during warm-up
        
    inference_timer.active = False
    for i in range(100):
        frames = conn.read_frame()
        if frames:
            im, metadatas = preprocess(frames)
            raw_output = inferencer(im)
            _ = post_processor(raw_output, metadatas)

    logger.info("Warm-up complete. Starting benchmarking...")

    # Enable timing for inference steps
    inference_timer.active = True
    attempts = 0
    while attempts < frames_to_process:
        full_pipeline(conn, inferencer, post_processor)
        attempts += 1
        if attempts % 100 == 0:
            logger.info(f"Processed {attempts} frames...")
        
        if config.SHOW_VIDEO:
            if cv2.waitKey(1) >= 0:
                break

    p99_preprocess = np.percentile(inference_timer.latencies['preprocess'], 99)
    p99_inference = np.percentile(inference_timer.latencies['PyTorchInferencer.__call__'], 99)
    p99_postprocess = np.percentile(inference_timer.latencies['PostProcess.__call__'], 99)
    p99_full_pipeline = np.percentile(inference_timer.latencies['full_pipeline'], 99)

    jitter_preprocess = np.std(inference_timer.latencies['preprocess'])
    jitter_inference = np.std(inference_timer.latencies['PyTorchInferencer.__call__'])
    jitter_postprocess = np.std(inference_timer.latencies['PostProcess.__call__'])
    jitter_full_pipeline = np.std(inference_timer.latencies['full_pipeline'])

    logger.info("*************** Latency Benchmark Results ***************\n")

    logger.info("================ Average Latencies (ms) ================\n")

    logger.info(f"{'Preprocess':<20} {np.mean(inference_timer.latencies['preprocess']):.4f}")
    logger.info(f"{'Inference':<20} {np.mean(inference_timer.latencies['PyTorchInferencer.__call__']):.4f}")
    logger.info(f"{'Post-process':<20} {np.mean(inference_timer.latencies['PostProcess.__call__']):.4f}")
    logger.info(f"{'Full Pipeline':<20} {np.mean(inference_timer.latencies['full_pipeline']):.4f}\n")
    logger.info("================ Average Latencies (ms) ================\n")
    logger.info(f"{'Average FPS':<20} {1000 / np.mean(inference_timer.latencies['full_pipeline']):.2f} FPS\n")
    logger.info("================ 99th Percentile Latencies (ms) ================\n")

    logger.info(f"{'Preprocess':<20} {p99_preprocess:.4f}")
    logger.info(f"{'Inference':<20} {p99_inference:.4f}")
    logger.info(f"{'Post-process':<20} {p99_postprocess:.4f}")
    logger.info(f"{'Full Pipeline':<20} {p99_full_pipeline:.4f}\n")

    logger.info("================ Jitter (ms) ================\n")
    logger.info(f"{'Preprocess':<20} {jitter_preprocess:.4f}")
    logger.info(f"{'Inference':<20} {jitter_inference:.4f}")
    logger.info(f"{'Post-process':<20} {jitter_postprocess:.4f}")
    logger.info(f"{'Full Pipeline':<20} {jitter_full_pipeline:.4f}\n")

    cv2.destroyAllWindows()

def bench_mAP(inferencer: PyTorchInferencer, post_processor: PostProcess, calculator: MAPCalculator, iou_threshold: float = 0.5):
    '''
    Run the full inference pipeline in a loop to benchmark mAP (mean Average Precision) of the model's predictions.
    This function collects predictions and ground truth data to calculate mAP after processing a set number of frames.
    '''
    # Placeholder for mAP benchmarking logic
    calculator.predict(batch_size=8)
    mAP50 = calculator.calculate_mAP(iou_threshold=iou_threshold)
    return mAP50


def bench_hardware(conn: CameraConnection, inferencer: PyTorchInferencer, post_processor: PostProcess, base_ram: float, frames_to_process: int = 1000):
    '''
    Run the full inference pipeline in a loop to benchmark hardware metrics such as GPU utilization, power consumption, temperature, and RAM usage.
    This function uses the HardwareMonitor to collect data during inference and logs summary statistics after completion.
    '''

    monitor = HardwareMonitor()
    monitor.start()

    logger.info("Starting warm-up phase...")

        # Warm up the model with 100 frames before starting timed inference
        # Reuse a single PostProcess instance to avoid allocation overhead during warm-up
        
    inference_timer.active = False
    for i in range(100):
        frames = conn.read_frame()
        if frames:
            im, metadatas = preprocess(frames)
            raw_output = inferencer(im)
            _ = post_processor(raw_output, metadatas)

    logger.info("Warm-up complete. Starting benchmarking...")

    # Enable timing for inference steps
    inference_timer.active = True
    attempts = 0
    while attempts < frames_to_process:
        full_pipeline(conn, inferencer, post_processor)
        attempts += 1

        if attempts % 100 == 0:
            logger.info(f"Processed {attempts} frames...")

        if config.SHOW_VIDEO:
            if cv2.waitKey(1) >= 0:
                break

    monitor.stop()

    stats = monitor.summary()
    
    logger.info("================ Hardware Stats ================\n")

    logger.info(f"{'GPU Util Avg':<20} {stats['gpu_util_avg']:.2f} %")
    logger.info(f"{'GPU Util Peak':<20} {stats['gpu_util_peak']:.2f} %\n")

    logger.info(f"{'Power Avg':<20} {stats['power_avg']/1000:.2f} W")
    logger.info(f"{'Power Peak':<20} {stats['power_peak']/1000:.2f} W\n")

    logger.info(f"{'Temperature Peak':<20} {stats['temp_peak']:.2f} °C")
    logger.info(f"{'RAM Total Peak':<20} {stats['ram_peak']:.2f} MB")
    logger.info(f"{'RAM Dynamic Peak':<20} {stats['ram_peak'] - base_ram:.2f} MB\n")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    
    ram_start = get_ram()
    

    inferencer = PyTorchInferencer(model_path="yolo11n.yaml", weights_path="pure_weight.pt", device=config.TARGET_DEVICE)
    
    if args.mode == 'map':
        post_processor = PostProcess(conf_threshold=0.001)
        calculator = MAPCalculator(inferencer=inferencer, post_processor=post_processor)
        
        for iou_threshold in np.arange(0.5, 1, 0.05):
            bench_mAP(inferencer, post_processor, calculator, iou_threshold=iou_threshold)
        calculator.print_mAPs()
    
    else:
        post_processor = PostProcess()
        with CameraConnection(camera_index=args.camera,
                              input_video_path='17431598-hd_1920_1080_60fps.mp4',
                              output_resolution=(1920, 1080) if args.camera is None else (640, 640)) as conn:

            if args.mode == 'run':
                just_run(conn, inferencer, post_processor, frames_to_process=args.frames)
            elif args.mode == 'latency':
                bench_latencies(conn, inferencer, post_processor, frames_to_process=args.frames)
            elif args.mode == 'hardware':
                bench_hardware(conn, inferencer, post_processor, ram_start, frames_to_process=args.frames)

    model_size = os.path.getsize("pure_weight.pt") / (1024 * 1024)
    logger.info(f"{'Model Size':<20} {model_size:.2f} MB")
    