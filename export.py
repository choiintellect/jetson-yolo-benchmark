import torch
import logging
import onnx

from ultralytics.nn.tasks import DetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model_path = 'yolo11n.yaml'
    onnx_path = 'yolo11n.onnx'
    weights_path = 'pure_weight.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = DetectionModel(cfg=model_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.fuse().eval().float().to(device=device)
        logger.debug(f"Model loaded successfully from {model_path} on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model. Check paths and device compatibility. Model path: {model_path}, Device: {device}") from e

    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input':{0:'batch_size', 2:'height', 3:'width'},'output':{0:'batch_size'}},
            opset_version=13,
            do_constant_folding=True,
        )
        logger.info(f"Model exported to ONNX format at: {onnx_path}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        raise RuntimeError(f"Failed to export model to ONNX. Check export parameters and compatibility. ONNX path: {onnx_path}") from e

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model at {onnx_path} is valid.")
    except Exception as e:
        logger.error(f"Error validating ONNX model: {e}")
        raise RuntimeError(f"Failed to validate ONNX model. Check the exported file for issues. ONNX path: {onnx_path}") from e
    
    import onnx

    model = onnx.load("yolo11n.onnx")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        # 형상 정보 출력
        dim = input.type.tensor_type.shape.dim
        print(f"Shape: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in dim]}")