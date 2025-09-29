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

"""
ITS (Intelligent Transportation Systems) Evaluator for Cosmos-RL.

This module combines the inference and scoring logic from the original ITS evaluation
into a single, integrated pipeline that works with the cosmos-rl CLI structure.
"""

import json
import logging as log
import os
import time
import glob
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial
import concurrent.futures
from tqdm import tqdm

import attrs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from qwen_vl_utils import process_vision_info

from cosmos_rl.utils.tao_status_logger import log_tao_status
from cosmos_rl.utils.logging import logger

# Set up image processing
Image.MAX_IMAGE_PIXELS = 933120000
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Check if debug model is enabled
DEBUG_MODEL: bool = os.getenv("DEBUG_MODEL", "0") == "1"

# Constants
COMPONENT_NAME = "Cosmos-RL ITS Evaluation"
DIRECTION_STRAIGHT = "going straight"
DIRECTION_LEFT = "turning left"
DIRECTION_RIGHT = "turning right"

# Import model and tokenizer based on DEBUG_MODEL flag
if DEBUG_MODEL:
    from tools.eval_utils.dummy_model import DummyModel, DummyTokenizer, SamplingParams
    LLM = DummyModel
    Processor = DummyTokenizer
else:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    Processor = AutoProcessor


@attrs.define(slots=False)
class ITSInputStructure:
    """Input structure for ITS evaluation tasks."""
    datasource: str
    media_id: str
    question: str
    question_idx: int
    media_paths: List[str]
    media_mode: str
    correct_answer: str
    prompt: Optional[Union[str, List[Dict[str, Any]]]] = None

    @classmethod
    def from_dict(cls, datasource: str, qa_pair: Dict[str, Any]) -> "ITSInputStructure":
        """Create ITSInputStructure from a question-answer pair dictionary."""
        return cls(
            datasource=datasource,
            media_id=qa_pair["media_id"],
            question=qa_pair['conversations'][1]["content"],
            correct_answer=qa_pair['conversations'][2]["content"],
            question_idx=qa_pair['id'],
            media_paths=qa_pair["media_paths"],
            media_mode=qa_pair["media_mode"],
            prompt=qa_pair['conversations'][:-1],
        )


@attrs.define(slots=False)
class ITSOutputStructure:
    """Output structure for ITS evaluation results."""
    datasource: str
    video_id: str
    correct_answer: str
    output_json_fname: str
    prompt: str = ""
    answer: str = ""
    reasoning: str = ""
    full_response: str = ""
    is_correct: bool = False


class ITSEvaluator:
    """
    Integrated ITS evaluator that combines inference and scoring.
    
    This class handles the complete evaluation pipeline:
    1. Loading and preparing datasets
    2. Running model inference
    3. Computing and reporting metrics
    4. Integrating with TAO status callbacks
    """

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False):
        """
        Initialize the ITS evaluator with configuration.
        
        Args:
            config: Evaluation configuration dictionary
            enable_lora: Whether to enable LoRA model merging
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.eval_config = config.get("evaluation", {})
        self.gen_config = config.get("generation", {})
        self.vision_config = config.get("vision", {})
        self.datasets = {"eval_set_name": config["dataset"]}
        self.enable_lora = enable_lora
        
        self.model = None
        self.processor = None
        
    def _load_model(self) -> Tuple[Any, Any]:
        """Load the model and processor."""
        log.info("Loading model and processor...")
        start_time = time.time()
        
        model_name = self.model_config.get("model_name")
        tokenizer_model_name = self.model_config.get("tokenizer_model_name", "qwen2.5-vl-7b")
        dtype = self.model_config.get("dtype", "bfloat16")
        tp_size = self.model_config.get("tp_size", 1)
        max_length = self.model_config.get("max_length", 128000)
        
        if DEBUG_MODEL:
            log.warning("DEBUG_MODEL is enabled. Using DummyModel and DummyTokenizer.")
            model = DummyModel()
            processor = DummyTokenizer()
        else:
            # Handle LoRA merging if enabled (from CLI flag or config)
            enable_lora_config = self.model_config.get("enable_lora", False)
            if self.enable_lora or enable_lora_config:
                model_name = self._merge_lora_model(model_name)
                log.info(f"LoRA merging enabled. Using merged model: {model_name}")
            
            model, processor = self._define_model(
                tokenizer_model_name, model_name, dtype, tp_size, max_length
            )
        
        log.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model, processor
    
    def _merge_lora_model(self, lora_path: str) -> str:
        """
        Merge LoRA weights with base model.
        
        Args:
            lora_path: Path to the LoRA model directory
            
        Returns:
            Path to the merged model directory
        """
        log.info(f"Merging LoRA model: {lora_path}")
        
        # Check if already merged
        merged_path = lora_path.replace("safetensors", "merged")
        if os.path.exists(merged_path) and os.path.isdir(merged_path):
            log.info(f"Merged model already exists at: {merged_path}")
            return merged_path
        
        try:
            # Import required libraries for LoRA merging
            from transformers import Qwen2_5_VLForConditionalGeneration
            from peft import PeftModel
            import torch
            
            # Get base model path from config or use default
            base_model_path = self.model_config.get("base_model_path")
            
            log.info(f"Loading base model: {base_model_path}")
            log.info(f"Loading LoRA adapter: {lora_path}")
            
            # Load base model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, 
                torch_dtype="auto"
            )
            
            # Load LoRA adapter
            peft_model = PeftModel.from_pretrained(model, lora_path)
            
            # Merge and unload
            log.info("Merging LoRA weights with base model...")
            merged_model = peft_model.merge_and_unload()
            
            # Save merged model
            os.makedirs(merged_path, exist_ok=True)
            log.info(f"Saving merged model to: {merged_path}")
            merged_model.save_pretrained(merged_path)
            
            # Clean up GPU memory
            del model, peft_model, merged_model
            torch.cuda.empty_cache()
            
            log.info(f"LoRA merging completed successfully: {merged_path}")
            
            # Log LoRA merge success to TAO
            log_tao_status(
                data={
                    "lora_merge_status": "success",
                    "merged_model_path": merged_path
                },
                component_name=COMPONENT_NAME
            )
            
            return merged_path
            
        except Exception as e:
            log.error(f"LoRA merging failed: {e}")
            log.warning(f"Falling back to original model path: {lora_path}")
            
            # Log LoRA merge failure to TAO
            log_tao_status(
                data={
                    "lora_merge_status": "failed",
                    "lora_merge_error": str(e)
                },
                component_name=COMPONENT_NAME
            )
            
            return lora_path
    
    def _define_model(
        self,
        tokenizer_model_name: str,
        model_name: str,
        dtype: str,
        tp_size: int,
        max_length: int = 128000,
    ) -> Tuple[Any, Any]:
        """Define and load the language model and processor."""
        # Import evaluation utilities
        from cosmos_rl.evaluation.utils.model_download import download_checkpoint, download_tokenizer
        
        if os.path.isabs(model_name) and os.path.exists(model_name):
            checkpoint_output_dir = model_name
        else:
            hf_cache_dir = os.environ.get(
                "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            )
            checkpoint_output_dir = os.path.join(hf_cache_dir, model_name)
            os.makedirs(checkpoint_output_dir, exist_ok=True)

        # Download checkpoint and tokenizer if needed
        download_checkpoint(model_name, checkpoint_output_dir)
        download_tokenizer(tokenizer_model_name, checkpoint_output_dir)

        log.info("Using VLLM backend.")
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        
        llm = LLM(
            model=checkpoint_output_dir,
            tokenizer=checkpoint_output_dir,
            trust_remote_code=True,
            dtype=dtype,
            limit_mm_per_prompt={"video": 10, "image": 10},
            tensor_parallel_size=tp_size,
            max_seq_len_to_capture=16384,
            max_model_len=max_length,
        )

        processor = AutoProcessor.from_pretrained(
            checkpoint_output_dir, max_length=max_length
        )

        return llm, processor
    
    def _make_all_tasks(
        self,
        results_output_folder: Path,
        answer_type: str,
        total_shard: int,
        shard_id: int,
    ) -> Tuple[List[ITSInputStructure], List[ITSOutputStructure]]:
        """Gather all evaluation tasks from datasets."""
        input_tasks = []
        output_results = []
        
        qa_pairs = []
        for datasource_name, datasource_config in self.datasets.items():
            log.info(f"Gathering tasks from dataset: {datasource_name}")

            media_dir = datasource_config.get("media_dir", None)
            annotation_path = datasource_config.get("annotation_path")
            system_prompt = datasource_config.get(
                "system_prompt", 
                ""
            )

            if answer_type == "reasoning" and '<think>' not in system_prompt:
                system_prompt += ("Answer the question with provided options in the following format: "
                                "\\n<think>\\nyour reasoning\\n</think> <answer>\\nyour answer\\n</answer>.")

            log.info(f"System prompt: {system_prompt}")
            
            # Check if media directory exists
            if media_dir and not os.path.exists(media_dir):
                log.error(f"Media path does not exist: {media_dir}")
                continue

            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
                
                for item in annotations:
                    # Clean question text
                    question = re.sub(r"(\\n)?</?(image|video)>(\\n)?", "", 
                                    item['conversations'][0]['value']).strip()
                    
                    conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": item['conversations'][1]['value']}
                    ]

                    # Handle media paths
                    images = item.get("image", None) or item.get("images", None)
                    videos = item.get("video", None)

                    if images:
                        if isinstance(images, str):
                            images = [images]
                        relative_media_paths = images
                        media_mode = 'image'
                    elif videos:
                        if isinstance(videos, str):
                            videos = [videos]
                        relative_media_paths = videos
                        media_mode = 'video'
                    else:
                        log.error(f"No media paths found for item: {item}")
                        continue

                    if media_dir:
                        media_paths = [os.path.join(media_dir, path) for path in relative_media_paths]
                    else:
                        media_paths = relative_media_paths
                        
                    qa_pairs.append({
                        "datasource": datasource_name,  # Add the datasource name
                        "media_id": relative_media_paths[0],
                        "id": item['id'],
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                        "conversations": conversation
                    })

        # Shard the tasks
        shard_qa_pairs = qa_pairs[shard_id::total_shard]
        log.info(f"Sharding {len(qa_pairs)} tasks into {total_shard} shards, "
                f"shard {shard_id} has {len(shard_qa_pairs)} tasks.")
        
        for qa_pair in shard_qa_pairs:
            output_json_fname = results_output_folder / qa_pair["datasource"] / f"{qa_pair['media_id']}.json"
            output_json_fname.parent.mkdir(parents=True, exist_ok=True)
            
            input_task = ITSInputStructure.from_dict(qa_pair["datasource"], qa_pair)
            output_result = ITSOutputStructure(
                datasource=qa_pair["datasource"],
                video_id=qa_pair['media_id'],
                correct_answer=qa_pair['conversations'][-1]["content"],
                output_json_fname=str(output_json_fname),
                prompt="",
            )
            
            input_tasks.append(input_task)
            output_results.append(output_result)

        return input_tasks, output_results
    
    def _prepare_model_inputs_parallel(
        self,
        input_tasks: List[ITSInputStructure],
        num_processes: int,
    ) -> List[Any]:
        """Prepare model inputs for tasks in parallel."""
        if not input_tasks:
            log.info("No input tasks to prepare model inputs for.")
            return []

        num_workers = min(num_processes, len(input_tasks))
        log.info(f"Preparing model inputs in parallel for {len(input_tasks)} tasks "
                f"using {num_workers} threads.")

        worker_fn = partial(
            self._prepare_single_model_input,
            processor=self.processor,
            vision_config=self.vision_config,
        )

        processed_inputs_with_index = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(worker_fn, task): i for i, task in enumerate(input_tasks)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Preparing model inputs",
            ):
                idx = future_to_idx[future]
                try:
                    model_input = future.result()
                    if model_input is not None:
                        processed_inputs_with_index.append((idx, model_input))
                except Exception as e:
                    log.error(f"Error preparing model input for task {idx}: {e}. Skipping task.")

        processed_inputs_with_index.sort(key=lambda x: x[0])
        inputs = [inp for idx, inp in processed_inputs_with_index]

        if len(inputs) < len(input_tasks):
            log.warning(f"Successfully prepared inputs for {len(inputs)} out of {len(input_tasks)} tasks.")

        return inputs
    
    def _prepare_single_model_input(
        self,
        input_task: ITSInputStructure,
        processor: Any,
        vision_config: Dict[str, Any],
    ) -> Optional[Any]:
        """Prepare input data for a single model inference task."""
        # Add video/image information to the user message content
        if (len(input_task.prompt) > 1
            and input_task.prompt[1]["role"] == "user"
            and isinstance(input_task.prompt[1]["content"], str)):
            
            content = []
            media_mode = input_task.media_mode
            for media_path in input_task.media_paths:
                video_content = {
                    "type": media_mode,
                    media_mode: media_path,
                }
                # Add vision config parameters
                for k, v in vision_config.items():
                    video_content[k] = v
                content.append(video_content)
            
            content.append({
                "type": "text",
                "text": input_task.prompt[1]["content"]
            })
            input_task.prompt[1]["content"] = content

        # Apply chat template
        processed_text_prompt = processor.apply_chat_template(
            input_task.prompt, tokenize=False, add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            input_task.prompt, return_video_kwargs=True
        )

        if not video_inputs and not image_inputs:
            log.error(f"No video or image inputs found for task: media_id={input_task.media_id}, "
                     f"question_idx={input_task.question_idx}. Cannot prepare model input.")
            return None

        if video_inputs:
            model_input = {
                "prompt": processed_text_prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs,
            }
        else:
            model_input = {
                "prompt": processed_text_prompt,
                "multi_modal_data": {"image": image_inputs},
            }

        log.debug(f"Prepared model input for task: media_id={input_task.media_id}, "
                 f"question_idx={input_task.question_idx}")
        return model_input
    
    def _run_model(
        self,
        inputs: List[str],
        input_tasks: List[ITSInputStructure],
        output_results: List[ITSOutputStructure],
        answer_type: str,
    ) -> None:
        """Run the model on inputs and process outputs."""
        # Get generation parameters
        max_tokens = self.gen_config.get("max_tokens", 1024)
        temperature = self.gen_config.get("temperature", 0)
        repetition_penalty = self.gen_config.get("repetition_penalty", 1.0)
        presence_penalty = self.gen_config.get("presence_penalty", 0.0)
        frequency_penalty = self.gen_config.get("frequency_penalty", 0.0)
        seed = self.eval_config.get("seed", 1)
        
        stop_token_id = self.processor.tokenizer.eos_token_id

        # Configure sampling parameters
        if answer_type == "letter":
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
                stop_token_ids=[stop_token_id],
                top_k=1,
                seed=seed,
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stop_token_ids=[stop_token_id],
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
            )

        log.info(f"Generating outputs for {len(inputs)} tasks using VLLM...")
        list_of_requestoutput = self.model.generate(inputs, sampling_params)
        log.info(f"Finished VLLM generation. Received {len(list_of_requestoutput)} outputs.")

        # Process outputs
        for i, (requestoutput, input_task, output_result) in enumerate(
            zip(list_of_requestoutput, input_tasks, output_results, strict=False)
        ):
            output_text = requestoutput.outputs[0].text
            
            # Parse based on answer type
            if answer_type == "letter":
                answer, reasoning = self._parse_letter_response(output_text)
            elif answer_type == "reasoning":
                answer, reasoning = self._parse_reasoning_response(output_text)
            else:
                answer = output_text
                reasoning = ""

            # Store results
            output_result.prompt = input_task.prompt
            output_result.reasoning = reasoning
            output_result.answer = answer
            output_result.full_response = output_text
            output_result.is_correct = (answer.lower() == input_task.correct_answer.lower())
    
    def _parse_letter_response(self, response: str) -> Tuple[str, str]:
        """Parse letter-format response."""
        # Simple implementation - can be enhanced
        return response.strip()[:1], ""
    
    def _parse_reasoning_response(self, response: str) -> Tuple[str, str]:
        """Parse reasoning-format response."""
        # Extract answer and reasoning from <think> and <answer> tags
        reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()
        
        return answer, reasoning
    
    def _save_results_parallel(
        self, 
        output_results: List[ITSOutputStructure], 
        num_processes: int
    ) -> None:
        """Save results to JSON files in parallel."""
        if not output_results:
            return

        num_workers = min(num_processes, len(output_results))
        log.info(f"Saving results in parallel using {num_workers} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._save_single_result, result)
                for result in output_results
            ]
            
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Saving results"
            ):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error saving result: {e}")
    
    def _save_single_result(self, result: ITSOutputStructure) -> None:
        """Save a single result to JSON file."""
        result_data = {
            "datasource": result.datasource,
            "video_id": result.video_id,
            "correct_answer": result.correct_answer,
            "answer": result.answer,
            "reasoning": result.reasoning,
            "full_response": result.full_response,
            "is_correct": result.is_correct,
        }
        
        os.makedirs(os.path.dirname(result.output_json_fname), exist_ok=True)
        with open(result.output_json_fname, 'w') as f:
            json.dump([result_data], f, indent=2)
    
    def _evaluate_directionality(self, result_path: Path) -> Dict[str, Any]:
        """
        Evaluate directionality metrics from results.
        
        This is the scoring logic from the original score.py.
        """
        log.info("Computing directionality metrics...")
        
        results = glob.glob(str(result_path / "eval_set_name" / "**" / "*.json"), recursive=True)
        log.info(f"Found {len(results)} result files.")
        
        correct_count = 0
        total_count = 0
        result_dict = {
            DIRECTION_STRAIGHT: {"correct": 0, "total": 0},
            DIRECTION_LEFT: {"correct": 0, "total": 0},
            DIRECTION_RIGHT: {"correct": 0, "total": 0}
        }
        
        word_classes = [DIRECTION_STRAIGHT, DIRECTION_LEFT, DIRECTION_RIGHT]
        confusion_matrix = np.zeros((len(word_classes), len(word_classes)), dtype=int)
        word_to_idx = {w: i for i, w in enumerate(word_classes)}
        
        for result_file in results:
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                
                for item in data:
                    total_count += 1
                    gt = item["correct_answer"].lower()
                    response = item["answer"].lower()
                    
                    # Determine ground truth type
                    if 'straight' in gt:
                        gt_type = DIRECTION_STRAIGHT
                    elif 'right' in gt:
                        gt_type = DIRECTION_RIGHT
                    elif 'left' in gt:
                        gt_type = DIRECTION_LEFT
                    else:
                        continue
                    
                    result_dict[gt_type]["total"] += 1
                    
                    # Determine response type
                    if 'straight' in response:
                        response_type = DIRECTION_STRAIGHT
                    elif 'right' in response:
                        response_type = DIRECTION_RIGHT
                    elif 'left' in response:
                        response_type = DIRECTION_LEFT
                    else:
                        response_type = "unknown"
                        # For confusion matrix, treat unknown as a miss
                        continue
                    
                    if response_type == gt_type:
                        correct_count += 1
                        result_dict[gt_type]["correct"] += 1
                    
                    # Update confusion matrix (only for known responses)
                    if response_type in word_to_idx:
                        confusion_matrix[word_to_idx[gt_type], word_to_idx[response_type]] += 1
                        
            except Exception as e:
                log.error(f"Error processing result file {result_file}: {e}")
        
        # Calculate accuracies
        for k, v in result_dict.items():
            if v["total"] > 0:
                accuracy = v["correct"] / v["total"]
                result_dict[k]["accuracy"] = accuracy
            else:
                result_dict[k]["accuracy"] = 0.0
        
        overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
        result_dict["overall"] = {
            "correct": correct_count,
            "total": total_count,
            "accuracy": overall_accuracy
        }
        
        # Save results
        results_file = result_path / "directionality_score.json"
        with open(results_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        # Create and save confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=word_classes, yticklabels=word_classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(result_path / "directionality_confusion_matrix.png")
        plt.close()
        
        log.info("Directionality evaluation completed:")
        log.info(f"  Overall accuracy: {overall_accuracy:.4f} ({correct_count}/{total_count})")
        for category, metrics in result_dict.items():
            if category != "overall":
                acc = metrics.get("accuracy", 0.0)
                correct = metrics.get("correct", 0)
                total = metrics.get("total", 0)
                log.info(f"  {category}: {acc:.4f} ({correct}/{total})")
        
        return result_dict
    
    def run_evaluation(
        self,
        results_dir: Path,
        skip_saved: bool = False,
        limit: int = -1,
        total_shard: int = 1,
        shard_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            results_dir: Directory to save results
            skip_saved: Whether to skip already saved results
            limit: Limit number of tasks (for debugging)
            total_shard: Total number of shards
            shard_id: Current shard ID
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        
        # Load model
        self.model, self.processor = self._load_model()
        
        # Get evaluation parameters
        answer_type = self.eval_config.get("answer_type", "freeform")
        num_processes = self.eval_config.get("num_processes", 40)
        
        # Create results directory structure
        save_folder = self.model_config.get("save_folder", None)
        if save_folder:
            results_output_dir = results_dir / save_folder
        else:
            model_name = self.model_config.get("model_name", "unknown_model")
            results_output_dir = results_dir / Path(model_name).name / answer_type
        
        results_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Gather tasks
        log.info("Gathering evaluation tasks...")
        input_tasks, output_results = self._make_all_tasks(
            results_output_dir, answer_type, total_shard, shard_id
        )
        log.info(f"Gathered {len(input_tasks)} tasks")
        
        # Log progress to TAO
        lora_status_data = {
            "evaluation_phase": "task_gathering",
            "total_tasks": len(input_tasks),
            "shard_id": shard_id,
            "total_shards": total_shard,
            "lora_enabled": self.enable_lora or self.model_config.get("enable_lora", False)
        }
        
        log_tao_status(
            data=lora_status_data,
            component_name=COMPONENT_NAME
        )
        
        # Step 2: Skip saved results if requested
        if skip_saved:
            log.info("Checking for saved results...")
            # Simple implementation - can be enhanced
            filtered_tasks = []
            filtered_results = []
            for task, result in zip(input_tasks, output_results):
                if not os.path.exists(result.output_json_fname):
                    filtered_tasks.append(task)
                    filtered_results.append(result)
            input_tasks, output_results = filtered_tasks, filtered_results
            log.info(f"Tasks remaining after skipping saved: {len(input_tasks)}")
        
        # Apply limit if specified
        if limit > 0 and len(input_tasks) > limit:
            input_tasks = input_tasks[:limit]
            output_results = output_results[:limit]
            log.info(f"Limited tasks to {len(input_tasks)} for debugging")
        
        if not input_tasks:
            log.info("No tasks to evaluate. Exiting.")
            return {"overall": {"accuracy": 0.0, "total": 0, "correct": 0}}
        
        # Step 3: Prepare model inputs
        log.info("Preparing model inputs...")
        inputs = self._prepare_model_inputs_parallel(input_tasks, num_processes)
        log.info(f"Prepared {len(inputs)} model inputs")
        
        # Log progress to TAO
        log_tao_status(
            data={
                "evaluation_phase": "input_preparation",
                "prepared_inputs": len(inputs)
            },
            component_name=COMPONENT_NAME
        )
        
        # Step 4: Run model inference
        log.info("Running model inference...")
        inference_start = time.time()
        self._run_model(inputs, input_tasks, output_results, answer_type)
        inference_time = time.time() - inference_start
        log.info(f"Model inference completed in {inference_time:.2f} seconds")
        
        # Log progress to TAO
        log_tao_status(
            data={
                "evaluation_phase": "inference",
                "inference_time_seconds": inference_time,
                "tasks_processed": len(input_tasks)
            },
            component_name=COMPONENT_NAME
        )
        
        # Step 5: Save results
        log.info("Saving results...")
        self._save_results_parallel(output_results, num_processes)
        
        # Step 6: Compute metrics
        log.info("Computing evaluation metrics...")
        metrics = self._evaluate_directionality(results_output_dir)
        
        total_time = time.time() - start_time
        log.info(f"Complete evaluation pipeline finished in {total_time:.2f} seconds")
        
        return metrics


def main():
    """
    Main entry point for cosmos-rl-evaluate command.
    
    This function provides a command-line interface for the ITS evaluator
    that can be called directly or through the pyproject.toml script entry point.
    """
    from cosmos_rl.evaluation.evaluate import main as evaluate_main
    evaluate_main()
