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

from cosmos_rl.utils.wfm.io.easy_io.handlers.base import BaseFileHandler
from cosmos_rl.utils.wfm.io.easy_io.handlers.byte_handler import (
    ByteHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.csv_handler import CsvHandler
from cosmos_rl.utils.wfm.io.easy_io.handlers.gzip_handler import (
    GzipHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.imageio_video_handler import (
    ImageioVideoHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.json_handler import (
    JsonHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.jsonl_handler import (
    JsonlHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.np_handler import (
    NumpyHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.pandas_handler import (
    PandasHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.pickle_handler import (
    PickleHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.pil_handler import PILHandler
from cosmos_rl.utils.wfm.io.easy_io.handlers.tarfile_handler import (
    TarHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.torch_handler import (
    TorchHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.torchjit_handler import (
    TorchJitHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.trimesh_handler import (
    TrimeshHandler,
)
from cosmos_rl.utils.wfm.io.easy_io.handlers.txt_handler import TxtHandler
from cosmos_rl.utils.wfm.io.easy_io.handlers.yaml_handler import (
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
