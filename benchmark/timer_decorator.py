from typing import Callable, Any
import torch
import time
import logging
from functools import wraps
from config import TIMER, TARGET_DEVICE

logger = logging.getLogger(__name__)

class Timer:
    """Decorator class for measuring execution time on CPU and GPU.

    This class provides precise timing for both CPU (using time.perf_counter)
    and GPU (using torch.cuda.Event). It also supports a 'Null Decorator' 
    pattern to eliminate overhead in production.

    Attributes:
        enabled (bool): Whether to enable timing. If False, the decorator is a no-op.
        
        active (bool): Runtime flag to toggle timing. Defaults to True.

        latencies (dict): A dictionary to store latency measurements for different functions.
            Keys are function names (e.g., 'preprocess', 'PyTorchInferencer.__call__', 'PostProcess.__call__', 'full_pipeline') and values are lists of recorded latencies in milliseconds.
    Methods:
        __call__: Wraps the target function and handles timing logic.

    Example:
        timer = Timer(enabled=True)

        @timer(device='cuda')
        def inference_step(model, input_tensor):
            return model(input_tensor)
    """
    def __init__(self, enabled: bool = TIMER):
        self.enabled = enabled
        self.active = True
        self.latencies = {
            'preprocess': [],
            'PyTorchInferencer.__call__': [],
            'PostProcess.__call__': [],
            'full_pipeline': []
        }

    def __call__(self, device: str = TARGET_DEVICE, get_log_per: int = 0) -> Callable[..., Any]:
        '''
         Wraps the target function to measure its execution time on the specified device.

         Args:
             device (str): The device to measure time on ('cpu' or 'cuda').

             get_log_per (int): Frequency of logging (in terms of number of calls). If 0, logging is disabled.

         Returns:
             Callable[..., Any]: The decorated function (wrapper) if enabled,
                 otherwise the original function.
        '''
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Wrap the function to measure its elapsed time.

            If 'enabled' is False, this returns the original function directly.
            If 'enabled' is True, it returns a wrapper that measures execution time on the specified device.

            Args:
                func (Callable[..., Any]): The function to be decorated. 
                    Can be any function with arbitrary arguments and return types.

            Returns:
                Callable[..., Any]: The decorated function (wrapper) if enabled,
                    otherwise the original function.
            """
            if not self.enabled:
                return func
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.active:
                    return func(*args, **kwargs)
                elapsed_time = None
                fname = func.__qualname__
                # Measure execution time based on GPU
                if device == 'cuda' and torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    result = func(*args, **kwargs)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)

                    

                # Measure execution time based on CPU
                else:
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    end = time.perf_counter()
                    elapsed_time = (end - start) * 1000  # Convert to milliseconds

                if fname in self.latencies:
                    self.latencies[fname].append(elapsed_time)

                if get_log_per > 0 and len(self.latencies[fname]) % get_log_per == 0:
                    logger.info(f'[{fname:^30}] \t Latency: {elapsed_time:.4f}ms')

                return result
            return wrapper
        return decorator

inference_timer = Timer()