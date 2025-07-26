from cosmos_rl.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Optional
import torch
import torch.nn.functional as F
from cosmos_rl.utils.util import retry
from cosmos_rl.policy.config import Config
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

IGNORE_LABEL_ID = -100


class HFVLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for the HF VLMs for SFT and RL training.
    """

    Payload = List[Dict[str, Any]]

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        self.hf_processor = retry(AutoProcessor.from_pretrained)(
            config.policy.model_name_or_path
        )

        hf_config = retry(AutoConfig.from_pretrained)(config.policy.model_name_or_path)

        image_token_id = getattr(hf_config, "image_token_id", None) or getattr(
            hf_config.vision_config, "image_token_id", None
        )
        if image_token_id is None:
            image_token_id = getattr(hf_config, "image_token_index", None) or getattr(
                hf_config.vision_config, "image_token_index", None
            )

        self.image_token_id = image_token_id
        # if image_token_id is None:
        #     self.image_token = None

        video_token_id = getattr(hf_config, "video_token_id", None) or getattr(
            hf_config.vision_config, "video_token_id", None
        )
        if video_token_id is None:
            video_token_id = getattr(hf_config, "video_token_index", None) or getattr(
                hf_config.vision_config, "video_token_index", None
            )
        if video_token_id is None:
            self.video_token = None
            self.video_token_id = None
        self.vision_ids = [self.image_token_id, self.video_token_id]
        self.hf_config = hf_config

    def get_rollout_input(self, sample: Payload) -> Any:
        """
        This VL data packer only accepts the conversation data format.
        check https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat for more details.

        example:
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        It is user's responsibility to ensure the conversation format is correct
          and multi-media files involved in conversation are accessible.
        """
        assert all(
            isinstance(x, dict) and "role" in x and "content" in x for x in sample
        ), "All samples should be in conversation format, but got: {}".format(sample)

        # Here we need to convert the conversation format to the format required by vllm
        prompt = self.hf_processor.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(sample)

        if video_inputs:
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs},
            }
        elif image_inputs:
            return {
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            }
        else:
            return {
                "prompt": prompt,
            }

    def _replace_assistant_content(
        self,
        token_ids: List[int],
        label_ids: List[int],
        pad_token_id: int,
        eos_token_id: int,
        replacement_ids: List[int],
        pad_run_length: int = 10,
    ) -> List[int]:
        """
        Find the first run of exactly `pad_run_length` pad_token_id's in token_ids,
        replace that run with replacement_ids, and return the new list.
        If no such run is found, returns the original list unchanged.
        """
        n = len(token_ids)
        target_run = [pad_token_id] * pad_run_length

        # find the start index of the first matching run
        for i in range(n - pad_run_length + 1):
            if token_ids[i : i + pad_run_length] == target_run:
                # splice in the replacement
                if (
                    len(token_ids) > i + pad_run_length
                    and token_ids[i + pad_run_length] == eos_token_id
                ):
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + [eos_token_id]
                        + label_ids[i + pad_run_length + 1 :]
                    )
                else:
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + label_ids[i + pad_run_length :]
                    )
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def _process_single_sample(
        self, conversation: "HFVLMDataPacker.Payload"
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_content = []
            messages = conversation["messages"]
            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"]
                    new_content = content.copy()
                    if isinstance(new_content, str):
                        assistant_content.append(new_content)
                        new_content = pad_token * pad_run_length
                    elif isinstance(new_content, dict):
                        assert "text" in new_content, f"text not in content: {content}"
                        assistant_content.append(new_content["text"])
                        new_content["text"] = pad_token * pad_run_length
                    elif isinstance(content, list):
                        for i, item in enumerate(content):
                            if isinstance(item, dict):
                                assert "text" in item, f"text not in content: {item}"
                                assistant_content.append(item["text"])
                                new_content[i]["text"] = pad_token * pad_run_length
                            else:
                                raise ValueError(
                                    f"Unsupported content type: {type(item)}"
                                )
                    else:
                        raise ValueError(f"Unsupported content type: {type(content)}")
                    message["content"] = new_content

            text = self.hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            images = conversation["images"]
            # image_inputs, video_inputs = process_vision_info(conversation)
            kwarg = {
                "padding": True,
                "return_tensors": "pt",
                "images": [images],
            }

            inputs = self.hf_processor(
                text=[text],
                **kwarg,
            )
            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_content:
                replacement_ids = self.tokenizer.encode(
                    assistant_content, add_special_tokens=False
                )

                replaced, input_ids, label_ids = self._replace_assistant_content(
                    input_ids,
                    label_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    replacement_ids=replacement_ids,
                    pad_run_length=pad_run_length,
                )
                if not replaced:
                    raise ValueError("No assistant content to replace")
                if len(input_ids) != len(label_ids):
                    raise ValueError(
                        f"input_ids and label_ids should have the same length, but got {len(input_ids)} and {len(label_ids)}"
                    )
        except Exception as e:
            print(f"Error processing sample: {e}, please fix to ensure SFT works")
            raise e

        result_dict = {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
        if "pixel_values_videos" in inputs:
            result_dict["pixel_values_videos"] = inputs["pixel_values_videos"]
            if "video_grid_thw" in inputs:
                result_dict["video_grid_thw"] = inputs["video_grid_thw"]
            else:
                result_dict["video_grid_thw"] = None
            result_dict["second_per_grid_ts"] = torch.tensor(
                inputs["second_per_grid_ts"], dtype=torch.float
            )
            result_dict["pixel_values_videos_lengths_per_sample"] = inputs[
                "pixel_values_videos"
            ].shape[0]

        if "pixel_values" in inputs:
            result_dict["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                result_dict["image_grid_thw"] = inputs["image_grid_thw"]
            else:
                result_dict["image_grid_thw"] = None
            result_dict["pixel_values_lengths_per_sample"] = inputs[
                "pixel_values"
            ].shape[0]
        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values_videos = [x["pixel_values_videos"] for x in processed_samples]
        video_grid_thw = [x["video_grid_thw"] for x in processed_samples]
        second_per_grid_ts = [x["second_per_grid_ts"] for x in processed_samples]
        pixel_values_images = [x["pixel_values_images"] for x in processed_samples]
        image_grid_thw = [x["image_grid_thw"] for x in processed_samples]
        pixel_values_videos_lengths_per_sample = [
            x["pixel_values_videos_lengths_per_sample"] for x in processed_samples
        ]
        pixel_values_images_lengths_per_sample = [
            x["pixel_values_images_lengths_per_sample"] for x in processed_samples
        ]

        if all([x is not None for x in pixel_values_videos]):
            assert all(
                [x is not None for x in pixel_values_videos]
            ), "pixel_values_videos should not be None"
            max_len = max([x.shape[0] for x in pixel_values_videos])
            for i in range(len(pixel_values_videos)):
                pixel_values_videos[i] = pixel_values_videos[i].unsqueeze(0)
                assert (
                    pixel_values_videos[i].ndim == 3
                ), f"pixel_values_videos[i].ndim: {pixel_values_videos[i].ndim}"
                pixel_values_videos[i] = F.pad(
                    pixel_values_videos[i],
                    (0, 0, 0, max_len - pixel_values_videos[i].shape[1]),
                )
            pixel_values_videos = torch.cat(pixel_values_videos, dim=0)
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
            second_per_grid_ts = torch.cat(second_per_grid_ts, dim=0)
        else:
            # TODO(jiaxin): handle the case when there is mixed input: some with video, some without video
            assert all(
                [x is None for x in pixel_values_videos]
            ), "pixel_values_videos should be None"
            pixel_values_videos = None
            video_grid_thw = None
            second_per_grid_ts = None
            pixel_values_videos_lengths_per_sample = None

        if all([x is not None for x in pixel_values_images]):
            max_len = max([x.shape[0] for x in pixel_values_images])
            for i in range(len(pixel_values_images)):
                pixel_values_images[i] = pixel_values_images[i].unsqueeze(0)
                assert (
                    pixel_values_images[i].ndim == 3
                ), f"pixel_values_images[i].ndim: {pixel_values_images[i].ndim}"
                pixel_values_images[i] = F.pad(
                    pixel_values_images[i],
                    (0, 0, 0, max_len - pixel_values_images[i].shape[1]),
                )
            pixel_values_images = torch.cat(pixel_values_images, dim=0)
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            # TODO(jiaxin): handle the case when there is mixed input: some with image, some without image
            assert all(
                [x is None for x in pixel_values_images]
            ), "pixel_values_images should be None"
            pixel_values_images = None
            image_grid_thw = None
            pixel_values_images_lengths_per_sample = None

        # Shape description:
        #
        # pixel_values_[videos/images]: (BATCH_SIZE, N_PATCH, HIDDEN_SIZE)
        # [video/image]_grid_thw: (BATCH_SIZE, 3)
        # second_per_grid_ts: (BATCH_SIZE, 1)
        # pixel_values_[videos/images]_lengths_per_sample: (BATCH_SIZE, 1)
        batch = {}
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos
            batch["video_grid_thw"] = video_grid_thw
            batch["second_per_grid_ts"] = second_per_grid_ts
            batch["pixel_values_videos_lengths_per_sample"] = torch.tensor(
                pixel_values_videos_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        if pixel_values_images is not None:
            batch["pixel_values_images"] = pixel_values_images
            batch["image_grid_thw"] = image_grid_thw
            batch["pixel_values_images_lengths_per_sample"] = torch.tensor(
                pixel_values_images_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        # Pad the input_ids, logprob_masks
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id]
                * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID]
                    * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )
        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len]
                + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )

        assert len(batch["input_ids"]) == len(
            batch["logprob_masks"]
        ), "The length of input_ids, logprob_masks should be the same"

        return batch

    def get_policy_input(
        self,
        sample: "HFVLMDataPacker.Payload",
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        # assert all(
        #     isinstance(x, dict) and "role" in x and "content" in x for x in sample
        # ), "All samples should be in conversation format, but got: {}".format(sample)

        x = self._process_single_sample(sample)

        return_dict = {}
        if "pixel_values_videos" in x:
            return_dict["pixel_values_videos"] = x["pixel_values_videos"]
            return_dict["video_grid_thw"] = x["video_grid_thw"]
            return_dict["second_per_grid_ts"] = x["second_per_grid_ts"]
            return_dict["pixel_values_videos_lengths_per_sample"] = x[
                "pixel_values_videos_lengths_per_sample"
            ]
        else:
            return_dict["pixel_values_videos"] = None
            return_dict["video_grid_thw"] = None
            return_dict["second_per_grid_ts"] = None
            return_dict["pixel_values_videos_lengths_per_sample"] = None

        if "pixel_values_images" in x:
            return_dict["pixel_values_images"] = x["pixel_values_images"]
            return_dict["image_grid_thw"] = x["image_grid_thw"]
            return_dict["pixel_values_images_lengths_per_sample"] = x[
                "pixel_values_images_lengths_per_sample"
            ]
        else:
            return_dict["pixel_values_images"] = None
            return_dict["image_grid_thw"] = None
            return_dict["pixel_values_images_lengths_per_sample"] = None

        # Common fields
        input_ids = x["input_ids"]
        completion_ids = []
        if rollout_output:
            completion_ids = self.tokenizer(rollout_output).input_ids  # Don't pad yet

        return_dict["input_ids"] = input_ids + completion_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0]
        )

        # TODO(jiaxin): this is special for SFT, will be removed in ``policy_collate_fn``
        return_dict["label_ids"] = x["label_ids"]
        return return_dict

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        return max([len(x["input_ids"]) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        for x in processed_samples:
            if "label_ids" in x:
                del x["label_ids"]
        return self._collate_fn(processed_samples, computed_max_len)

    def sft_process_sample(self, sample: "HFVLMDataPacker.Payload") -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        return self.get_policy_input(sample)

    def sft_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return max([len(x["input_ids"]) for x in processed_samples])

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        # Reuse the RL collate minibatch function
        model_inputs: Dict[str, Any] = self._collate_fn(
            processed_samples, computed_max_len
        )
        del model_inputs["logprob_masks"]
        # Mask the loss on vision padding tokens
        if self.vision_ids is not None:
            assert isinstance(self.vision_ids, list)
            for vision_id in self.vision_ids:
                if vision_id is not None:
                    model_inputs["label_ids"][
                        model_inputs["label_ids"] == vision_id
                    ] = ignore_label_id

        return model_inputs
