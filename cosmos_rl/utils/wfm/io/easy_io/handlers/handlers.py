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

"""Consolidated file handlers for easy_io."""

import csv
import gzip
import json
import pickle
import tarfile
from abc import ABCMeta, abstractmethod
from io import BytesIO, StringIO
from typing import IO, Any, Dict, Optional, Tuple, Union

import imageio
import imageio.v3 as iio_v3
import numpy as np
import pandas as pd
import trimesh

from cosmos_rl.utils.logging import logger

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import yaml

    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
except ImportError:
    yaml = None


class BaseFileHandler(metaclass=ABCMeta):
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


class ByteHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        if not isinstance(obj, bytes):
            raise TypeError(f"Expected bytes but got {type(obj)}")
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        if not isinstance(obj, bytes):
            raise TypeError(f"Expected bytes but got {type(obj)}")
        return obj


class CsvHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        reader = csv.reader(file)
        return list(reader)

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        writer = csv.writer(file)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        output = StringIO()
        writer = csv.writer(output)
        if not all(isinstance(row, list) for row in obj):
            raise ValueError("Each row must be a list")
        writer.writerows(obj)
        return output.getvalue()


class GzipHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: BytesIO, **kwargs):
        with gzip.GzipFile(fileobj=file, mode="rb") as f:
            return pickle.load(f)

    def dump_to_fileobj(self, obj: Any, file: BytesIO, **kwargs):
        with gzip.GzipFile(fileobj=file, mode="wb") as f:
            pickle.dump(obj, f)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class ImageioVideoHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(
        self, file: IO[bytes], format: str = "mp4", mode: str = "rgb", **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load video from a file-like object using imageio.v3 with specified format and color mode.

        Parameters:
            file (IO[bytes]): A file-like object containing video data.
            format (str): Format of the video file (default 'mp4').
            mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

        Returns:
            tuple: A tuple containing an array of video frames and metadata about the video.
        """
        file.seek(0)

        # The plugin argument in v3 replaces the format argument in v2
        plugin = kwargs.pop("plugin", "pyav")

        # Load all frames at once using v3 API
        video_frames = iio_v3.imread(file, plugin=plugin, **kwargs)

        # Handle grayscale conversion if needed
        if mode == "gray":
            import cv2

            if len(video_frames.shape) == 4:  # (frames, height, width, channels)
                gray_frames = []
                for frame in video_frames:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    gray_frame = np.expand_dims(
                        gray_frame, axis=2
                    )  # Keep dimensions consistent
                    gray_frames.append(gray_frame)
                video_frames = np.array(gray_frames)

        # Extract metadata
        # Note: iio_v3.imread doesn't return metadata directly like v2 did
        # We need to extract it separately
        file.seek(0)
        metadata = self._extract_metadata(file, plugin=plugin)

        return video_frames, metadata

    def _extract_metadata(
        self, file: IO[bytes], plugin: str = "pyav"
    ) -> Dict[str, Any]:
        """
        Extract metadata from a video file.

        Parameters:
            file (IO[bytes]): File-like object containing video data.
            plugin (str): Plugin to use for reading.

        Returns:
            dict: Video metadata.
        """
        try:
            # Create a generator to read frames and metadata
            metadata = iio_v3.immeta(file, plugin=plugin)

            # Add some standard fields similar to v2 metadata format
            if "fps" not in metadata and "duration" in metadata:
                # Read the first frame to get shape information
                file.seek(0)
                first_frame = iio_v3.imread(file, plugin=plugin, index=0)
                metadata["size"] = first_frame.shape[1::-1]  # (width, height)
                metadata["source_size"] = metadata["size"]

                # Create a consistent metadata structure with v2
                metadata["plugin"] = plugin
                if "codec" not in metadata:
                    metadata["codec"] = "unknown"
                if "pix_fmt" not in metadata:
                    metadata["pix_fmt"] = "unknown"

                # Calculate nframes if possible
                if "fps" in metadata and "duration" in metadata:
                    metadata["nframes"] = int(metadata["fps"] * metadata["duration"])
                else:
                    metadata["nframes"] = float("inf")

            return metadata

        except Exception:
            # Fallback to basic metadata
            return {
                "plugin": plugin,
                "nframes": float("inf"),
                "codec": "unknown",
                "fps": 30.0,  # Default values
                "duration": 0,
                "size": (0, 0),
            }

    def dump_to_fileobj(
        self,
        obj: np.ndarray | torch.Tensor,
        file: IO[bytes],
        format: str = "mp4",  # pylint: disable=redefined-builtin
        fps: int = 17,
        quality: int = 5,
        ffmpeg_params=None,
        **kwargs,
    ):
        """
        Save an array of video frames to a file-like object using imageio.

        Parameters:
            obj (Union[np.ndarray, torch.Tensor]): An array of frames to be saved as video.
            file (IO[bytes]): A file-like object to which the video data will be written.
            format (str): Format of the video file (default 'mp4').
            fps (int): Frames per second of the output video (default 17).
            quality (int): Quality of the video (0-10, default 5).
            ffmpeg_params (list): Additional parameters to pass to ffmpeg.

        """
        if isinstance(obj, torch.Tensor):
            assert obj.dtype == torch.uint8, "Tensor must be of type uint8"
            obj = obj.cpu().numpy()
        h, w = obj.shape[1:-1]

        # Default ffmpeg params that ensure width and height are set
        default_ffmpeg_params = ["-s", f"{w}x{h}"]

        # Use provided ffmpeg_params if any, otherwise use defaults
        final_ffmpeg_params = (
            ffmpeg_params if ffmpeg_params is not None else default_ffmpeg_params
        )

        mimsave_kwargs = {
            "fps": fps,
            "quality": quality,
            "macro_block_size": 1,
            "ffmpeg_params": final_ffmpeg_params,
            "output_params": ["-f", "mp4"],
        }
        # Update with any other kwargs
        mimsave_kwargs.update(kwargs)
        logger.debug(f"mimsave_kwargs: {mimsave_kwargs}")

        imageio.mimsave(file, obj, format, **mimsave_kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj)} is unsupported for json dump")


class JsonHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return json.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("default", set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("default", set_default)
        return json.dumps(obj, **kwargs)


class JsonlHandler(BaseFileHandler):
    """Handler for JSON lines (JSONL) files."""

    def load_from_fileobj(self, file: IO[bytes]):
        """Load JSON objects from a newline-delimited JSON (JSONL) file object.

        Returns:
            A list of Python objects loaded from each JSON line.
        """
        data = []
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines if any
            data.append(json.loads(line))
        return data

    def dump_to_fileobj(self, obj: IO[bytes], file, **kwargs):
        """Dump a list of objects to a newline-delimited JSON (JSONL) file object.

        Args:
            obj: A list (or iterable) of objects to dump line by line.
        """
        kwargs.setdefault("default", set_default)
        for item in obj:
            file.write(json.dumps(item, **kwargs) + "\n")

    def dump_to_str(self, obj, **kwargs):
        """Dump a list of objects to a newline-delimited JSON (JSONL) string."""
        kwargs.setdefault("default", set_default)
        lines = [json.dumps(item, **kwargs) for item in obj]
        return "\n".join(lines)


class NumpyHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: IO[bytes], **kwargs) -> Any:
        """
        Load a NumPy array from a file-like object.

        Parameters:
            file (IO[bytes]): The file-like object containing the NumPy array data.
            **kwargs: Additional keyword arguments passed to `np.load`.

        Returns:
            numpy.ndarray: The loaded NumPy array.
        """
        return np.load(file, **kwargs)

    def load_from_path(self, filepath: str, **kwargs) -> Any:
        """
        Load a NumPy array from a file path.

        Parameters:
            filepath (str): The path to the file to load.
            **kwargs: Additional keyword arguments passed to `np.load`.

        Returns:
            numpy.ndarray: The loaded NumPy array.
        """
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj: np.ndarray, **kwargs) -> str:
        """
        Serialize a NumPy array to a string in binary format.

        Parameters:
            obj (np.ndarray): The NumPy array to serialize.
            **kwargs: Additional keyword arguments passed to `np.save`.

        Returns:
            str: The serialized NumPy array as a string.
        """
        with BytesIO() as f:
            np.save(f, obj, **kwargs)
            return f.getvalue()

    def dump_to_fileobj(self, obj: np.ndarray, file: IO[bytes], **kwargs):
        """
        Dump a NumPy array to a file-like object.

        Parameters:
            obj (np.ndarray): The NumPy array to dump.
            file (IO[bytes]): The file-like object to which the array is dumped.
            **kwargs: Additional keyword arguments passed to `np.save`.
        """
        np.save(file, obj, **kwargs)

    def dump_to_path(self, obj: np.ndarray, filepath: str, **kwargs):
        """
        Dump a NumPy array to a file path.

        Parameters:
            obj (np.ndarray): The NumPy array to dump.
            filepath (str): The file path where the array should be saved.
            **kwargs: Additional keyword arguments passed to `np.save`.
        """
        with open(filepath, "wb") as f:
            np.save(f, obj, **kwargs)


class PandasHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pd.read_csv(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        obj.to_csv(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError("PandasHandler does not support dumping to str")


class PickleHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)


class PILHandler(BaseFileHandler):
    format: str
    str_like = False

    def load_from_fileobj(
        self,
        file: IO[bytes],
        fmt: str = "pil",
        size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Load an image from a file-like object and return it in a specified format.

        Args:
            file (IO[bytes]): A file-like object containing the image data.
            fmt (str): The format to convert the image into. Options are \
                'numpy', 'np', 'npy', 'type' (all return numpy arrays), \
                    'pil' (returns PIL Image), 'th', 'torch' (returns a torch tensor).
            size (Optional[Union[int, Tuple[int, int]]]): The new size of the image as a single integer \
                or a tuple of (width, height). If specified, the image is resized accordingly.
            **kwargs: Additional keyword arguments that can be passed to conversion functions.

        Returns:
            Image data in the format specified by `fmt`.

        Raises:
            IOError: If the image cannot be loaded or processed.
            ValueError: If the specified format is unsupported.
        """
        try:
            img = Image.open(file)
            img.load()  # Explicitly load the image data
            if size is not None:
                if isinstance(size, int):
                    size = (
                        size,
                        size,
                    )  # create a tuple if only one integer is provided
                img = img.resize(size, Image.ANTIALIAS)

            # Return the image in the requested format
            if fmt in ["numpy", "np", "npy"]:
                return np.array(img, **kwargs)
            if fmt == "pil":
                return img
            if fmt in ["th", "torch"]:
                import torch

                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(img, **kwargs))
                # Convert image from HxWxC to CxHxW
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
                return img_tensor
            raise ValueError(
                "Unsupported format. Supported formats are 'numpy', 'np', 'npy', 'pil', 'th', and 'torch'."
            )
        except Exception as e:
            raise IOError(f"Unable to load image: {e}") from e

    def dump_to_fileobj(self, obj, file: IO[bytes], **kwargs):
        if "format" not in kwargs:
            kwargs["format"] = self.format
        kwargs["format"] = (
            "JPEG" if self.format.lower() == "jpg" else self.format.upper()
        )
        obj.save(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class TarHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, mode="r|*", **kwargs):
        return tarfile.open(fileobj=file, mode=mode, **kwargs)

    def load_from_path(self, filepath, mode="r|*", **kwargs):
        return tarfile.open(filepath, mode=mode, **kwargs)

    def dump_to_fileobj(self, obj, file, mode="w", **kwargs):
        with tarfile.open(fileobj=file, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with tarfile.open(filepath, mode=mode) as tar:
            tar.add(obj, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class TorchHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return torch.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        torch.save(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class TorchJitHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return torch.jit.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        torch.jit.save(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class TrimeshHandler(BaseFileHandler):
    format: str
    str_like = False

    def load_from_fileobj(self, file: IO[bytes], **kwargs) -> trimesh.Trimesh:
        file = trimesh.load(file_obj=file, file_type=self.format)
        return file

    def dump_to_fileobj(self, obj, file: IO[bytes], **kwargs):
        obj.export(file_obj=file, file_type=self.format)
        return file

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError


class TxtHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        return obj


class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
