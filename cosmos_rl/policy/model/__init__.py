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

import importlib
import inspect
from pathlib import Path

# Import base classes and registry
from cosmos_rl.policy.model.base import ModelRegistry, BaseModel, WeightMapper

# Import DiffuserModel base class
from cosmos_rl.policy.model.diffusers import DiffuserModel

# Dictionary to store all imported models
__all__ = ["ModelRegistry", "BaseModel", "WeightMapper", "DiffuserModel"]


def _discover_model_classes(module, base_class):
    """Discover model classes that inherit from the specified base class in a module."""
    discovered_classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if it's a subclass of the base class (but not the base class itself)
        if obj != base_class and issubclass(obj, base_class):
            # Ensure the class is actually defined in this module, not imported
            if obj.__module__ == module.__name__:
                discovered_classes.append(name)
    return discovered_classes


def _auto_import_models():
    """Dynamically import all model modules from the model directory."""
    model_dir = Path(__file__).parent

    # Directories to exclude from auto-discovery
    exclude_dirs = {"__pycache__", "base", "diffusers", "vision_encoder", "wfm"}

    # ============================================================================
    # BaseModel imports (all top-level model directories except diffusers)
    # ============================================================================
    print("Auto-discovering BaseModel subclasses...")
    for item in model_dir.iterdir():
        if item.is_dir() and item.name not in exclude_dirs:
            module_name = item.name
            try:
                module = importlib.import_module(
                    f"cosmos_rl.policy.model.{module_name}"
                )
                # Discover all BaseModel subclasses in the module
                classes = _discover_model_classes(module, BaseModel)
                for class_name in classes:
                    if hasattr(module, class_name):
                        globals()[class_name] = getattr(module, class_name)
                        __all__.append(class_name)
                        print(
                            f"  [Model] Imported BaseModel: {class_name} from {module_name}"
                        )
            except ImportError as e:
                print(f"  [Model] Warning: Could not import module {module_name}: {e}")

    # ============================================================================
    # DiffuserModel imports (from diffusers/ subdirectory)
    # ============================================================================
    print("Auto-discovering DiffuserModel subclasses...")
    diffusers_dir = model_dir / "diffusers"
    if diffusers_dir.exists():
        for item in diffusers_dir.iterdir():
            if (
                item.is_file()
                and item.suffix == ".py"
                and item.stem not in ["__init__", "parallelize", "weight_mapper"]
            ):
                module_name = f"diffusers.{item.stem}"
                try:
                    module = importlib.import_module(
                        f"cosmos_rl.policy.model.{module_name}"
                    )
                    # Discover all DiffuserModel subclasses in the module
                    classes = _discover_model_classes(module, DiffuserModel)
                    for class_name in classes:
                        if hasattr(module, class_name):
                            globals()[class_name] = getattr(module, class_name)
                            __all__.append(class_name)
                            print(
                                f"  [Model] Imported DiffuserModel: {class_name} from {module_name}"
                            )
                except ImportError as e:
                    print(
                        f"  [Model] Warning: Could not import Diffusers module {module_name}: {e}"
                    )

    # ============================================================================
    # WFM imports (from wfm/models/ subdirectory)
    # only cosmos-policy for now, models under this folder are implemented in i4 fashion
    # ============================================================================
    print("Auto-discovering wfm subclasses...")
    wfm_dir = model_dir / "wfm" / "models"
    if wfm_dir.exists():
        for item in wfm_dir.iterdir():
            if (
                item.is_file()
                and item.suffix == ".py"
                and item.stem in ["cosmos_policy"]
            ):
                module_name = f"wfm.models.{item.stem}"
                try:
                    module = importlib.import_module(
                        f"cosmos_rl.policy.model.{module_name}"
                    )
                    classes = _discover_model_classes(module, BaseModel)
                    for class_name in classes:
                        if hasattr(module, class_name):
                            globals()[class_name] = getattr(module, class_name)
                            __all__.append(class_name)
                            print(
                                f"  [Model] Imported BaseModel: {class_name} from {module_name}"
                            )
                except ImportError as e:
                    print(
                        f"  [Model] Warning: Could not import WFM module {module_name}: {e}"
                    )


# Run auto-import on module load
_auto_import_models()
