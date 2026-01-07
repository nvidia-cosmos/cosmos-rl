import importlib
from functools import wraps
import warnings

if importlib.util.find_spec("xformers") is not None:
    # Fix xformers non-compatible, for environment with xformers installed
    # Silently make it unfindable when using importlib.util.find_spec
    # TODO (yy): may find a better solution
    _orig_find_spec = importlib.util.find_spec

    @wraps(_orig_find_spec)
    def blocked_find_spec(name, *args, **kwargs):
        if name == "xformers" or name.startswith("xformers."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = blocked_find_spec
    warnings.warn("xformers is not compatible with our framework, please uninstall it")

from diffusers import DiffusionPipeline

diffusers_config_fn = DiffusionPipeline.load_config

__all__ = ["diffusers_config_fn", "DiffusionPipeline"]
