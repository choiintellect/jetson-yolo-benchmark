import torch
import logging
import onnx

from ultralytics.nn.tasks import DetectionModel

from benchmark.timer_decorator import inference_timer
from config import TARGET_DEVICE
from benchmark.hardware_monitor import get_ram

logger = logging.getLogger(__name__)

class PyTorchInferencer:
    """YOLOv11 inference engine using pure PyTorch.

    This class handles the initialization of a YOLOv11 detection model, 
    applies hardware-level optimizations (fusion, channels_last), and 
    provides a callable interface for inference.

    Attributes:
        device (torch.device): The device on which the model resides (cpu/cuda).

        model (DetectionModel): The fused and optimized YOLOv11 model.

    Methods:
        __call__: Executes a single inference pass on a preprocessed tensor.

    Example:
        inferencer = PyTorchInferencer("yolo11n.pt", device='cuda')
        raw_output = inferencer(input_tensor)
    """
    def __init__(self, model_path: str = 'yolo11n.yaml', weights_path: str = 'pure_weight.pt', device: str = TARGET_DEVICE):
        """Initialize and optimize the DetectionModel.

        Loads the model architecture and weights, then applies fusion.

        Args:
            model_path (str): Path to the model config (.yaml file).

            weights_path (str): Path to the model weights (.pt file).

            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        logger.debug(f"Initializing PyTorchInferencer with model: {model_path}, weights: {weights_path}, device: {device}")

        # Load model architecture and map weight
        try:
            ram_start = get_ram()
            self.model = DetectionModel(cfg=model_path)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.model = self.model.fuse().eval().float().to(self.device)
            ram_end = get_ram()
            self.static_memory = ram_end - ram_start
            logger.info(f"Static Memory Loaded: {self.static_memory:.2f} MB")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model. Check paths and device compatibility. Model path: {model_path}, Weights path: {weights_path}, Device: {device}") from e

        logger.debug(f"Success loading model: {model_path} ({device})")

    @torch.inference_mode()
    @inference_timer(device=TARGET_DEVICE)
    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass on the input image tensor.

        This method takes a preprocessed 4D tensor and returns raw detections.

        Args:
            im (torch.Tensor): Preprocessed image tensor with shape (N, 3, 640, 640).

        Returns:
            torch.Tensor: Raw output tensor from the model's head.
                Shape is typically (N, 84, 8400) for YOLOv11.
        """
        head = self.model(im)[0]
        
        logger.debug(f"Model output shape: {head.shape}")
        return head
    
    def onnx_export(self, input_shape=(1, 3, 640, 640), onnx_path='yolo11n.onnx'):
        """Export the model to ONNX format.

        This method exports the PyTorch model to ONNX format for interoperability.

        Args:
            input_shape (tuple): The shape of the input tensor (N, C, H, W).

            onnx_path (str): The file path to save the ONNX model.
        """
        dummy_input = torch.randn(input_shape).to(self.device)


        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input':{0:'batch_size', 2:'height', 3:'width'},'output':{0:'batch_size'}},
            opset_version=13,
            do_constant_folding=True,
        )
        
        logger.info(f"Model exported to ONNX format at: {onnx_path}")

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model at {onnx_path} is valid.")
    


if __name__ == "__main__":
    inferencer = PyTorchInferencer()
    inferencer.onnx_export()