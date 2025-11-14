/*!
**************************************************************************************************
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/huggingface/transformers/blob/main/src/transformers/kernels/deformable_detr/cuda/ms_deform_attn_cuda.h

# Copyright (c) 2020 SenseTime.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight);

std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int64_t im2col_step);