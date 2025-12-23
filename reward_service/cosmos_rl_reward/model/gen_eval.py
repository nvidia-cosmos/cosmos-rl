# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions of this file are adapted from NVLabs DiffusionNFT (https://github.com/NVlabs/DiffusionNFT)
# GenEval evaluation framework from djghosh13 (https://github.com/djghosh13/geneval)

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps

from cosmos_rl_reward.handler.reward_base import BaseRewardHandler
from cosmos_rl_reward.handler.registry import RewardRegistry
from cosmos_rl_reward.utils.logging import logger


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index,
                self._image.size[::-1],
                mask.shape,
            )
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self._transform(image), 0)


@RewardRegistry.register()
class GenEvalReward(BaseRewardHandler):
    NEEDS_LATENT_DECODER = False
    reward_name = "gen_eval"

    def __init__(self, model_path: str = "", download_path: str = "", device: str = "cuda", dtype: str = "float32", **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # model_path is unused for gen_eval; paths are resolved purely from download_path
        self.model_path = model_path
        self.download_path = download_path
        # No per-run overrides; rely solely on download_path convention
        self.compute_geneval = None

    @classmethod
    def load_geneval(cls, DEVICE, config_path, ckpt_path, object_names_path=None, classnames=None):
        import open_clip
        from clip_benchmark.metrics import zeroshot_classification as zsc
        from mmdet.apis import inference_detector, init_detector

        zsc.tqdm = lambda it, *args, **kwargs: it

        def timed(fn):
            def wrapper(*args, **kwargs):
                startt = time.time()
                result = fn(*args, **kwargs)
                endt = time.time()
                logger.info(f"[gen_eval] Function {fn.__name__!r} executed in {endt - startt:.3f}s")
                return result

            return wrapper

        @timed
        def load_models():
            object_detector = init_detector(config_path, ckpt_path, device=DEVICE)
            clip_arch = "ViT-L-14"
            clip_model, _, transform = open_clip.create_model_and_transforms(
                clip_arch, pretrained="openai", device=DEVICE
            )
            tokenizer = open_clip.get_tokenizer(clip_arch)
            return object_detector, (clip_model, transform, tokenizer)

        COLORS = [
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "pink",
            "brown",
            "black",
            "white",
        ]
        COLOR_CLASSIFIERS = {}

        object_detector, (clip_model, transform, tokenizer) = load_models()
        if object_names_path and os.path.exists(object_names_path):
            with open(object_names_path) as f:
                classnames = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(
                f"[gen_eval] object_names.txt not found at {object_names_path}. "
                f"Please run setup to download it or provide 'object_names_path'."
            )

        THRESHOLD = 0.3
        COUNTING_THRESHOLD = 0.9
        MAX_OBJECTS = 16
        NMS_THRESHOLD = 1.0
        POSITION_THRESHOLD = 0.1

        def color_classification(image, bboxes, classname):
            if classname not in COLOR_CLASSIFIERS:
                COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                    clip_model,
                    tokenizer,
                    COLORS,
                    [
                        f"a photo of a {{c}} {classname}",
                        f"a photo of a {{c}}-colored {classname}",
                        f"a photo of a {{c}} object",
                    ],
                    str(DEVICE),
                )
            clf = COLOR_CLASSIFIERS[classname]
            dataloader = torch.utils.data.DataLoader(
                ImageCrops(image, bboxes, transform=transform), batch_size=16, num_workers=0
            )
            with torch.no_grad():
                pred, _ = zsc.run_classification(clip_model, clf, dataloader, str(DEVICE))
                return [COLORS[index.item()] for index in pred.argmax(1)]

        def compute_iou(box_a, box_b):
            area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
            i_area = area_fn(
                [
                    max(box_a[0], box_b[0]),
                    max(box_a[1], box_b[1]),
                    min(box_a[2], box_b[2]),
                    min(box_a[3], box_b[3]),
                ]
            )
            u_area = area_fn(box_a) + area_fn(box_b) - i_area
            return i_area / u_area if u_area else 0

        def relative_position(obj_a, obj_b):
            boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
            center_a, center_b = boxes.mean(axis=-2)
            dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
            offset = center_a - center_b
            revised_offset = (
                np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
            )
            if np.all(np.abs(revised_offset) < 1e-3):
                return set()
            dx, dy = revised_offset / np.linalg.norm(offset)
            relations = set()
            if dx < -0.5:
                relations.add("left of")
            if dx > 0.5:
                relations.add("right of")
            if dy < -0.5:
                relations.add("above")
            if dy > 0.5:
                relations.add("below")
            return relations

        def evaluate(image, objects, metadata):
            correct = True
            reason = []
            matched_groups = []
            for req in metadata.get("include", []):
                classname = req["class"]
                matched = True
                found_objects = objects.get(classname, [])[: req["count"]]
                if len(found_objects) < req["count"]:
                    correct = matched = False
                    reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
                else:
                    if "color" in req:
                        colors = color_classification(image, found_objects, classname)
                        if colors.count(req["color"]) < req["count"]:
                            correct = matched = False
                            reason.append(
                                f"expected {req['color']} {classname}>={req['count']}, found "
                                + f"{colors.count(req['color'])} {req['color']}; and "
                                + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                            )
                    if "position" in req and matched:
                        expected_rel, target_group = req["position"]
                        if matched_groups[target_group] is None:
                            correct = matched = False
                            reason.append(f"no target for {classname} to be {expected_rel}")
                        else:
                            for obj in found_objects:
                                for target_obj in matched_groups[target_group]:
                                    true_rels = relative_position(obj, target_obj)
                                    if expected_rel not in true_rels:
                                        correct = matched = False
                                        reason.append(
                                            f"expected {classname} {expected_rel} target, found "
                                            + f"{' and '.join(true_rels)} target"
                                        )
                                        break
                                if not matched:
                                    break
                if matched:
                    matched_groups.append(found_objects)
                else:
                    matched_groups.append(None)
            for req in metadata.get("exclude", []):
                classname = req["class"]
                if len(objects.get(classname, [])) >= req["count"]:
                    correct = False
                    reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
            return correct, "\n".join(reason)

        def evaluate_reward(image, objects, metadata):
            correct = True
            reason = []
            rewards = []
            matched_groups = []
            for req in metadata.get("include", []):
                classname = req["class"]
                matched = True
                found_objects = objects.get(classname, [])
                rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
                if len(found_objects) != req["count"]:
                    correct = matched = False
                    reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                    if "color" in req or "position" in req:
                        rewards.append(0.0)
                else:
                    if "color" in req:
                        colors = color_classification(image, found_objects, classname)
                        rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                        if colors.count(req["color"]) != req["count"]:
                            correct = matched = False
                            reason.append(
                                f"expected {req['color']} {classname}>={req['count']}, found "
                                + f"{colors.count(req['color'])} {req['color']}; and "
                                + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                            )
                    if "position" in req and matched:
                        expected_rel, target_group = req["position"]
                        if matched_groups[target_group] is None:
                            correct = matched = False
                            reason.append(f"no target for {classname} to be {expected_rel}")
                            rewards.append(0.0)
                        else:
                            for obj in found_objects:
                                for target_obj in matched_groups[target_group]:
                                    true_rels = relative_position(obj, target_obj)
                                    if expected_rel not in true_rels:
                                        correct = matched = False
                                        reason.append(
                                            f"expected {classname} {expected_rel} target, found "
                                            + f"{' and '.join(true_rels)} target"
                                        )
                                        rewards.append(0.0)
                                        break
                                if not matched:
                                    break
                            rewards.append(1.0)
                if matched:
                    matched_groups.append(found_objects)
                else:
                    matched_groups.append(None)
            reward = sum(rewards) / len(rewards) if rewards else 0
            return correct, reward, "\n".join(reason)

        def evaluate_image(image_pils, metadatas, only_strict):
            from mmdet.apis import inference_detector

            results = inference_detector(object_detector, [np.array(image_pil) for image_pil in image_pils])
            ret = []
            for result, image_pil, metadata in zip(results, image_pils, metadatas):
                bbox = result[0] if isinstance(result, tuple) else result
                segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                image = ImageOps.exif_transpose(image_pil)
                detected = {}
                confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
                for index, classname in enumerate(classnames):
                    ordering = np.argsort(bbox[index][:, 4])[::-1]
                    ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                    ordering = ordering[:MAX_OBJECTS].tolist()
                    detected[classname] = []
                    while ordering:
                        max_obj = ordering.pop(0)
                        detected[classname].append(
                            (
                                bbox[index][max_obj],
                                None if segm is None else segm[index][max_obj],
                            )
                        )
                        ordering = [
                            obj
                            for obj in ordering
                            if NMS_THRESHOLD == 1
                            or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                        ]
                    if not detected[classname]:
                        del detected[classname]
                is_strict_correct, score, reason = evaluate_reward(image, detected, metadata)
                if only_strict:
                    is_correct = False
                else:
                    is_correct, _ = evaluate(image, detected, metadata)
                ret.append(
                    {
                        "tag": metadata["tag"],
                        "prompt": metadata["prompt"],
                        "correct": is_correct,
                        "strict_correct": is_strict_correct,
                        "score": score,
                        "reason": reason,
                        "metadata": json.dumps(metadata),
                        "details": json.dumps(
                            {key: [box.tolist() for box, _ in value] for key, value in detected.items()}
                        ),
                    }
                )
            return ret

        @torch.no_grad()
        def compute_geneval(images, metadatas, only_strict=False):
            required_keys = [
                "single_object",
                "two_object",
                "counting",
                "colors",
                "position",
                "color_attr",
            ]
            scores = []
            strict_rewards = []
            grouped_strict_rewards = defaultdict(list)
            rewards = []
            grouped_rewards = defaultdict(list)
            results = evaluate_image(images, metadatas, only_strict=only_strict)
            for result in results:
                strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
                scores.append(result["score"])
                rewards.append(1.0 if result["correct"] else 0.0)
                tag = result["tag"]
                for key in required_keys:
                    if key != tag:
                        grouped_strict_rewards[key].append(-10.0)
                        grouped_rewards[key].append(-10.0)
                    else:
                        grouped_strict_rewards[tag].append(1.0 if result["strict_correct"] else 0.0)
                        grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
            return (
                scores,
                rewards,
                strict_rewards,
                dict(grouped_rewards),
                dict(grouped_strict_rewards),
            )

        return compute_geneval

    def _resolve_paths(self):
        base_dir = self.download_path
        ckpt_basename = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99"
        ckpt_path = os.path.join(base_dir, "reward_ckpts", f"{ckpt_basename}.pth")
        config_path = os.path.join(
            base_dir,
            "mmdetection",
            "configs",
            "mask2former",
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        )
        return config_path, ckpt_path

    def set_up(self):
        device = self.device if isinstance(self.device, str) else ("cuda" if torch.cuda.is_available() else "cpu")
        config_path, ckpt_path = self._resolve_paths()
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"[gen_eval] mmdet config not found at {config_path}. "
                f"Please run setup script or set 'download_path' correctly."
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[gen_eval] detector ckpt not found at {ckpt_path}. "
                f"Please run setup script or set 'download_path' correctly."
            )
        object_names_path = os.path.join(self.download_path, "reward_ckpts", "object_names.txt")
        self.compute_geneval = self.load_geneval(
            device, config_path, ckpt_path, object_names_path=object_names_path
        )

    def calculate_reward(self, images, metadata):
        st = time.time()
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(
                f"gen_eval expects 4D uint8 torch.Tensor in BHWC/NHWC layout (B,H,W,C); got type={type(images)} shape={getattr(images,'shape',None)} dtype={getattr(images,'dtype',None)}"
            )
        x = images.to(torch.uint8)
        imgs = [Image.fromarray(x[i].cpu().numpy()) for i in range(x.shape[0])]
        metas = metadata.get("video_infos")
        if not isinstance(metas, list) or len(metas) != len(imgs):
            metas = [metadata] * len(imgs)
        if getattr(self, "compute_geneval", None) is None:
            raise RuntimeError(
                "[gen_eval] Inferencer not initialized. set_up() was not called or failed; check initialization logs."
            )
        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = self.compute_geneval(
            imgs, metas, only_strict=bool(metadata.get("geneval_only_strict", False))
        )
        logger.info(f"[gen_eval] Score: {scores}")
        logger.info(f"[gen_eval] Reward: {rewards}")
        logger.info(f"[gen_eval] Strict reward: {strict_rewards}")
        logger.info(f"[gen_eval] Group reward: {group_rewards}")
        logger.info(f"[gen_eval] Group strict reward: {group_strict_rewards}")

        return {
            "scores": {
                "gen_eval_score": scores,
                "gen_eval_reward": rewards,
                "gen_eval_strict": strict_rewards,
                "gen_eval_group": group_rewards,
            },
            "input_info": metadata.get("input_info", {}),
            "duration": f"{time.time() - st:.2f}",
            "decoded_duration": metadata.get("decode_duration", "N/A"),
            "type": self.reward_name,
        }