import io
import os
import json
from collections import OrderedDict
from typing import Callable, Optional, List
import random
import boto3
from botocore.config import Config
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def make_s3_client_from_env():
    account_id = os.getenv("R2_ACCOUNT_ID")
    endpoint = os.getenv("R2_ENDPOINT") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "auto")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    return s3

class _LRU:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, k):
        if k in self._d:
            v = self._d.pop(k)
            self._d[k] = v
            return v
        return None

    def put(self, k, v):
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


def random_resize_min(img: Image.Image, min_size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    # scale so both sides >= min_size
    min_scale = max(min_size / w, min_size / h)
    # allow scale only if ≤ 1.0 (never upscale)
    if min_scale > 1.0:
        # if image is too small, force resize up
        scale = min_scale
    else:
        scale = random.uniform(min_scale, 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

def random_crop(img: Image.Image, crop_size: int) -> Image.Image:
    """
    Random crop of 512x512 region (assumes both sides >= 512).
    """
    w, h = img.size
    if w == crop_size and h == crop_size:
        return img
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    return img.crop((x, y, x + crop_size, y + crop_size))

def pil_scaled_then_crop(img: Image.Image, target_size: int) -> Image.Image:
    """
    Full pipeline:
      1. Random scale in [min_scale, 1.0] so min side >= 512
      2. Random 512x512 crop
    Returns PIL.Image.
    """
    img = random_resize_min(img, target_size)
    img = random_crop(img, target_size)
    return img

class R2ImageDataset(Dataset):
    """
    Random-access image dataset directly from Cloudflare R2 using boto3.
    Expects a manifest.json produced by the uploader; otherwise can enumerate by prefix.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        manifest_key: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_size: int = 128,
        image_extensions: Optional[List[str]] = None,
        s3_client=None,
        image_size: int = 512,
    ):
        """
        Args:
            bucket: R2 bucket name
            prefix: key prefix where your dataset lives (e.g., "datasets/" or "my_project/run1/")
            manifest_key: if provided, key to the manifest.json; if None, we list objects under prefix
            transform: optional torchvision-style transform
            cache_size: number of decoded PIL images to keep in an in-memory LRU cache
            image_extensions: filter keys by extension if no manifest is given
            s3_client: optional boto3 S3 client. If None, will construct from environment
        """
        self.bucket = bucket
        self.prefix = prefix if not prefix or prefix.endswith("/") else (prefix + "/")
        self.transform = transform
        self.s3 = s3_client or make_s3_client_from_env()
        self.cache = _LRU(cache_size)
        self.image_extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        self.image_size = image_size

        keys = self._try_load_manifest(manifest_key)
        assert manifest_key is not None
        try:
            for i in range(1000):
                manifest_key = self.prefix + f"manifest-{i}.json"
                keys.extend(self._try_load_manifest(manifest_key))
        except Exception:
            pass
        if keys is None:
            keys = self._list_image_keys()

        if not keys:
            raise RuntimeError("No images found in R2 at the specified location.")
        print(f"Found {len(keys)} images in R2 at the specified location.")
        self.keys = keys

    def _try_load_manifest(self, key: str):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = obj["Body"].read()
            manifest = json.loads(data)
            items = manifest.get("items", [])
            # Use exactly the keys recorded (preserves subfolder structure)
            return [it["key"] for it in items if "key" in it and self._is_image_key(it["key"])]
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def _list_image_keys(self):
        keys = []
        kwargs = {"Bucket": self.bucket, "Prefix": self.prefix, "MaxKeys": 1000}
        while True:
            resp = self.s3.list_objects_v2(**kwargs)
            contents = resp.get("Contents", [])
            for c in contents:
                k = c["Key"]
                if self._is_image_key(k):
                    keys.append(k)
            if resp.get("IsTruncated"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return keys

    def _is_image_key(self, k: str) -> bool:
        lower = k.lower()
        return any(lower.endswith(ext) for ext in self.image_extensions)

    def __len__(self):
        return len(self.keys)

    def _load_image_from_r2(self, key: str) -> Image.Image:
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        data = resp["Body"].read()  # bytes
        img = Image.open(io.BytesIO(data)).convert("RGB")
        self.cache.put(key, img)
        return img

    def __getitem__(self, index: int):
        key = self.keys[index]
        while True:
            try:
                img = self._load_image_from_r2(key)
                img = pil_scaled_then_crop(img, self.image_size)
                break
            except Exception as e:
                print(f"Error loading image {key}: {e}")
                index = random.randint(0, len(self.keys) - 1)
                key = self.keys[index]
                print(f"Retrying with new index {index}")
        if self.transform:
            img = self.transform(img)
        else:
            # conver from PIL to tensor
            img = TF.to_tensor(img)
        # If you have labels embedded in folder names, derive here (example below)
        # label = key.split("/")[-2]  # e.g., datasets/class_x/img123.jpg
        # return img, label
        return img

def make_loader(bucket: str, prefix: str = "", cache_size: int = 128, image_size: int = 512, batch_size: int = 64, num_workers: int = 0):
    manifest_key = prefix + "manifest.json"
    ds = R2ImageDataset(bucket=bucket, prefix=prefix, manifest_key=manifest_key, cache_size=cache_size, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

if __name__ == "__main__":
    # Expect the same env vars as the uploader
    bucket = os.getenv("R2_BUCKET")
    prefix = os.getenv("R2_PREFIX", "datasets/")  # where you uploaded
    manifest_key = prefix + "manifest.json"       # created by uploader

    ds = make_loader(bucket=bucket, prefix=prefix, cache_size=256, image_size=512, batch_size=64, num_workers=0)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)  # num_workers>0 is fine; boto3 is thread-safe per-client

    for batch in dl:
        # batch is a tensor of shape [B, C, H, W]
        print(f'batch: {batch}') 
        # save batch to disk
        for i in range(batch.shape[0]):
            img = TF.to_pil_image(batch[i])
            img.save(f'batch_{i}.png')
        break