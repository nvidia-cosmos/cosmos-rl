from cosmos_rl.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Optional, Tuple
import torch
from cosmos_rl.utils.util import retry
from cosmos_rl.policy.config import Config
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
import logging
import copy

IGNORE_LABEL_ID = -100


class Qwen2_5_VLM_DataPacker(DataPacker):
    """
    Data protocol & processing logic for the Qwen2.5 VLM for SFT and RL training.
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
        # disable qwen_vl_utils logger
        qwen_vl_utils_logger = logging.getLogger("qwen_vl_utils")
        qwen_vl_utils_logger.setLevel(logging.WARNING)
        qwen_vl_utils_logger.propagate = False

        hf_config = retry(AutoConfig.from_pretrained)(config.policy.model_name_or_path)
        if hasattr(hf_config, "image_token_id"):
            self.image_token_id = hf_config.image_token_id
        else:
            self.image_token = None
            self.image_token_id = None
        if hasattr(hf_config, "video_token_id"):
            self.video_token_id = hf_config.video_token_id
        else:
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
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            sample, return_video_kwargs=True
        )

        if video_inputs:
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs,
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

    def _get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        hf_config = self.hf_config
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        mrope_position_deltas = []
        second_per_grid_ts = (
            second_per_grid_ts.cpu() if second_per_grid_ts is not None else None
        )
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * hf_config.vision_config.tokens_per_second
                    )

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _process_single_sample(
        self,
        conversation: "Qwen2_5_VLM_DataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_content = []
            conversation = copy.deepcopy(conversation)
            for message in conversation:
                if message["role"] == "assistant":
                    content = message["content"]
                    if isinstance(content, str):
                        assistant_content.append(content)
                    elif isinstance(content, dict):
                        assert "text" in content, f"text not in content: {content}"
                        assistant_content.append(content["text"])
                    else:
                        raise ValueError(f"Unsupported content type: {type(content)}")
                    message["content"] = pad_token * pad_run_length

            prompt = self.hf_processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.hf_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
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
            "input_ids": copy.deepcopy(input_ids),
            "label_ids": copy.deepcopy(label_ids),
        }
        if "pixel_values_videos" in inputs:
            result_dict["pixel_values_videos"] = inputs["pixel_values_videos"]
            result_dict["video_grid_thw"] = inputs["video_grid_thw"]
            result_dict["second_per_grid_ts"] = torch.tensor(
                inputs["second_per_grid_ts"], dtype=torch.float
            )

        if "pixel_values" in inputs:
            result_dict["pixel_values"] = inputs["pixel_values"]
            result_dict["image_grid_thw"] = inputs["image_grid_thw"]

        # position_ids: (3, 1, seq_len)
        position_ids, _ = self._get_rope_index(
            input_ids=torch.tensor(input_ids).unsqueeze(0).clone(),
            image_grid_thw=torch.tensor(result_dict.get("image_grid_thw"))
            if "image_grid_thw" in result_dict
            else None,
            video_grid_thw=torch.tensor(result_dict.get("video_grid_thw"))
            if "video_grid_thw" in result_dict
            else None,
            second_per_grid_ts=torch.tensor(result_dict.get("second_per_grid_ts"))
            if "second_per_grid_ts" in result_dict
            else None,
            attention_mask=None,
        )
        result_dict["position_ids"] = position_ids.clone()
        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values_videos = [x["pixel_values_videos"] for x in processed_samples]
        video_grid_thw = [x["video_grid_thw"] for x in processed_samples]
        second_per_grid_ts = [x["second_per_grid_ts"] for x in processed_samples]
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_grid_thw = [x["image_grid_thw"] for x in processed_samples]
        if all([x is not None for x in pixel_values_videos]):
            assert all(
                [x is not None for x in pixel_values_videos]
            ), "pixel_values_videos should not be None"
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

        if all([x is not None for x in pixel_values]):
            pixel_values = torch.cat(pixel_values, dim=0)
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            # TODO(jiaxin): handle the case when there is mixed input: some with image, some without image
            assert all([x is None for x in pixel_values]), "pixel_values should be None"
            pixel_values = None
            image_grid_thw = None

        # Shape description:
        #
        # pixel_values_[videos/images]: (N_PATCH, HIDDEN_SIZE)
        # [video/image]_grid_thw: (BATCH_SIZE, 3)
        # second_per_grid_ts: (BATCH_SIZE, 1)
        batch = {}
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos
            batch["video_grid_thw"] = video_grid_thw
            batch["second_per_grid_ts"] = second_per_grid_ts

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
            batch["image_grid_thw"] = image_grid_thw

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

        padded_tensors = []
        for sample in processed_samples:
            pad_length = computed_max_len - sample["position_ids"].shape[2]
            padded_tensor = torch.nn.functional.pad(
                sample["position_ids"], (0, pad_length), "constant", 1
            )
            padded_tensors.append(padded_tensor)
        batch["position_ids"] = torch.cat(padded_tensors, dim=1)

        return batch

    def get_policy_input(
        self,
        sample: "Qwen2_5_VLM_DataPacker.Payload",
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        assert all(
            isinstance(x, dict) and "role" in x and "content" in x for x in sample
        ), "All samples should be in conversation format, but got: {}".format(sample)

        x = self._process_single_sample(
            sample, add_generation_prompt=add_generation_prompt
        )

        return_dict = {
            "position_ids": x["position_ids"],
        }
        if "pixel_values_videos" in x:
            return_dict["pixel_values_videos"] = x["pixel_values_videos"]
            return_dict["video_grid_thw"] = x["video_grid_thw"]
            return_dict["second_per_grid_ts"] = x["second_per_grid_ts"]
        else:
            return_dict["pixel_values_videos"] = None
            return_dict["video_grid_thw"] = None
            return_dict["second_per_grid_ts"] = None

        if "pixel_values" in x:
            return_dict["pixel_values"] = x["pixel_values"]
            return_dict["image_grid_thw"] = x["image_grid_thw"]
        else:
            return_dict["pixel_values"] = None
            return_dict["image_grid_thw"] = None

        # Common fields
        input_ids = x["input_ids"]
        completion_ids = []
        if rollout_output:
            completion_ids = self.tokenizer(rollout_output).input_ids
            # recompute position_ids
            # position_ids: (3, 1, seq_len)
            position_ids, _ = self._get_rope_index(
                input_ids=torch.tensor(input_ids + completion_ids).unsqueeze(0).clone(),
                image_grid_thw=torch.tensor(x.get("image_grid_thw"))
                if "image_grid_thw" in x
                else None,
                video_grid_thw=torch.tensor(x.get("video_grid_thw"))
                if "video_grid_thw" in x
                else None,
                second_per_grid_ts=torch.tensor(x.get("second_per_grid_ts"))
                if "second_per_grid_ts" in x
                else None,
                attention_mask=None,
            )
            return_dict["position_ids"] = position_ids.clone()
            return_dict["input_ids"] = input_ids + completion_ids
        else:
            return_dict["input_ids"] = input_ids

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

    def sft_process_sample(
        self, sample: "Qwen2_5_VLM_DataPacker.Payload"
    ) -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        return self.get_policy_input(sample, add_generation_prompt=False)

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
