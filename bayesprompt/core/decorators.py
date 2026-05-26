import time
import logging
import functools
import torch
import inspect
from typing import Callable, Any

logger = logging.getLogger(__name__)

def register_model(name: str = None):
    from .registry import MODELS
    return MODELS.register(name)

def register_dataset(name: str = None):
    from .registry import DATASETS
    return DATASETS.register(name)

def register_loss(name: str = None):
    from .registry import LOSSES
    return LOSSES.register(name)

def register_metric(name: str = None):
    from .registry import METRICS
    return METRICS.register(name)

def timeit(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

def log_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

def seeded(seed: int = 42):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .reproducibility import set_seed
            set_seed(seed)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def torch_no_grad(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

def ensure_frozen(module_name: str):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            module = getattr(self, module_name, None)
            if module is not None:
                for param in module.parameters():
                    if param.requires_grad:
                        raise RuntimeError(f"Parameter in {module_name} requires grad, but should be frozen!")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def rank_zero_only(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Simplified for non-DDP setup. In full DDP, check local_rank.
        return func(*args, **kwargs)
    return wrapper

def autocast_if_available(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def save_config(func: Callable):
    @functools.wraps(func)
    def wrapper(cfg, *args, **kwargs):
        # Assumes first arg is config object with a save method or similar
        return func(cfg, *args, **kwargs)
    return wrapper

def catch_oom(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory caught!")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e
            else:
                raise e
    return wrapper

def validate_shapes(arg=None):
    """
    Validates tensor shapes of kwargs.
    Supports both `@validate_shapes` and `@validate_shapes({'prompts': (None, 'num_classes', 'embed_dim')})`.
    """
    def decorator(func: Callable):
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if isinstance(arg, dict):
                for param_name, expected_shape in arg.items():
                    if param_name in bound_args.arguments:
                        val = bound_args.arguments[param_name]
                        if isinstance(val, torch.Tensor):
                            if len(val.shape) != len(expected_shape):
                                raise ValueError(f"Shape mismatch for {param_name}. Expected rank {len(expected_shape)}, got {len(val.shape)}")
                            for actual, expected in zip(val.shape, expected_shape):
                                if isinstance(expected, int) and actual != expected:
                                    raise ValueError(f"Shape mismatch for {param_name}. Expected {expected_shape}, got {val.shape}")
            return func(*args, **kwargs)
        return wrapper
        
    if callable(arg):
        return decorator(arg)
    return decorator

def requires_checkpoint(func: Callable):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'checkpoint_loaded') or not self.checkpoint_loaded:
            logger.warning(f"{func.__name__} called but no checkpoint loaded.")
        return func(self, *args, **kwargs)
    return wrapper

def experiment_entrypoint(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"--- Starting Experiment: {func.__name__} ---")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Experiment {func.__name__} failed: {str(e)}")
            raise e
        finally:
            logger.info(f"--- Finished Experiment: {func.__name__} ---")
    return wrapper
