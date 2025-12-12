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

import os
import dataclasses


@dataclasses.dataclass
class ItemDatasetConfig:
    path: str
    length: int


def get_itemdataset_option(
    name: str, text_embedding_type: str = "t5_xxl"
) -> ItemDatasetConfig:
    item_dataset_config = ITEMDATASET_OPTIONS[name]

    if text_embedding_type != "t5_xxl":
        # For all datasets other than T5_XXL, we save the dataset in the following path
        # {data_root}/ablation_text_embeddings/{text_embedding_type}/{dataset_name}
        dataset_path = item_dataset_config.path
        dataset_path_split = dataset_path.split("/")
        is_file = os.path.splitext(dataset_path)[1] != ""

        if is_file:
            # In case of a file, we have
            # {data_root}/ablation_text_embeddings/{text_embedding_type}/{dataset_name}/{filename.ext}
            new_dataset_path = (
                dataset_path_split[0:-2]
                + ["ablation_text_embeddings", f"{text_embedding_type}"]
                + dataset_path_split[-2:]
            )
        else:
            new_dataset_path = (
                dataset_path_split[0:-1]
                + ["ablation_text_embeddings", f"{text_embedding_type}"]
                + dataset_path_split[-1:]
            )

        new_dataset_path = "/".join(new_dataset_path)

        return ItemDatasetConfig(
            path=new_dataset_path, length=item_dataset_config.length
        )
    return item_dataset_config


# length must % 8 =0 to avoid mysterious hang bug of fsdp+CP!It is tested with cp4.
ITEMDATASET_OPTIONS = {}
