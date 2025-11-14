import torch
from torch.utils.data.dataset import Dataset

import os
import json
from PIL import Image, ImageOps
from typing import Any, Tuple, List, Optional


from .coco import COCO


class ODDataset(Dataset):
    """Base Object Detection Dataset Class."""

    def __init__(
        self,
        json_file: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        transforms=None,
    ):
        """Initialize the Object Detetion Dataset Class.

        Note that multiple loading of COCO type JSON files can lead to system memory OOM.
        In such case, use SerializedDatasetFromList.

        Args:
            json_file (str): json_file name to load the data.
            dataset_dir (str): dataset directory.
            transforms: augmentations to apply.
        """
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        with open(json_file, "r") as f:
            json_data = json.load(f)
        self.coco = COCO(json_data)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.label_map = self.coco.dataset["categories"]

    def _load_image(self, img_id: int) -> Image.Image:
        """Load image given image id.

        Args:
            img_id (int): image id to load.

        Returns:
            Loaded PIL Image.
        """
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        if not self.dataset_dir == "":
            img_path = os.path.join(self.dataset_dir, path)
            img = Image.open(img_path).convert("RGB")
            return_output = (ImageOps.exif_transpose(img), img_path)
        else:
            img = Image.open(path).convert("RGB")
            return_output = (ImageOps.exif_transpose(img), path)

        return return_output

    def _load_target(self, img_id: int) -> List[Any]:
        """Load target (annotation) given image id.

        Args:
            img_id (int): image id to load.

        Returns:
            Loaded COCO annotation list
        """
        return self.coco.loadAnns(self.coco.getAnnIds(img_id))

    def _process_image_target(
        self, image: Image.Image, target: List[Any], img_id: int
    ) -> Tuple[Any, Any]:
        """Process the image and target given image id.

        Args:
            image (PIL.Image): Loaded image given img_id.
            target (list): Loaded annotation given img_id.
            img_id (int): image id to load.

        Returns:
            (image, target): pre-processed image and target for the model.
        """
        width, height = image.size
        image_id = torch.tensor([img_id])

        boxes = [obj["bbox"] for obj in target]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = classes[keep]

        area = torch.tensor([obj["area"] for obj in target])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in target]
        )

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        return image, target

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """Get image, target, image_path given index.

        Args:
            index (int): index of the image id to load.

        Returns:
            (image, target, image_path): pre-processed image, target and image_path for the model.
        """
        img_id = self.ids[index]
        image, image_path = self._load_image(img_id)

        target = self._load_target(img_id)
        image, target = self._process_image_target(image, target, img_id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.ids)
