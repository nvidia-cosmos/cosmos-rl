# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

# Import all handlers and registry utilities from the consolidated handlers module
from cosmos_rl.utils.wfm.io.easy_io.handlers.handlers import (
    BaseFileHandler,
    ByteHandler,
    CsvHandler,
    GzipHandler,
    ImageioVideoHandler,
    JsonHandler,
    JsonlHandler,
    NumpyHandler,
    PandasHandler,
    PickleHandler,
    PILHandler,
    TarHandler,
    TorchHandler,
    TorchJitHandler,
    TrimeshHandler,
    TxtHandler,
    YamlHandler,
)


file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
    "tar": TarHandler(),
    "jit": TorchJitHandler(),
    "npy": NumpyHandler(),
    "txt": TxtHandler(),
    "csv": CsvHandler(),
    "pandas": PandasHandler(),
    "gz": GzipHandler(),
    "jsonl": JsonlHandler(),
    "byte": ByteHandler(),
}

for torch_type in ["pt", "pth", "ckpt"]:
    file_handlers[torch_type] = TorchHandler()
for img_type in ["jpg", "jpeg", "png", "bmp", "gif"]:
    file_handlers[img_type] = PILHandler()
    file_handlers[img_type].format = img_type
for mesh_type in ["ply", "stl", "obj", "glb"]:
    file_handlers[mesh_type] = TrimeshHandler()
    file_handlers[mesh_type].format = mesh_type
for video_type in ["mp4", "avi", "mov", "webm", "flv", "wmv"]:
    file_handlers[video_type] = ImageioVideoHandler()


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            f"handler must be a child of BaseFileHandler, not {type(handler)}"
        )
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not all([isinstance(item, str) for item in file_formats]):
        raise TypeError("file_formats must be a str or a list of str")
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):
    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
