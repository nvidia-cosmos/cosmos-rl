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
from typing import get_origin, get_args, Type, Any


def type_to_string(typ: Type) -> str:
    """type -> string"""
    if typ is type(None):
        return "NoneType"

    # generic type
    origin = get_origin(typ)
    if origin is not None:
        args = get_args(typ)
        origin_str = type_to_string(origin)
        if args:
            args_str = ", ".join(type_to_string(arg) for arg in args)
            return f"{origin_str}[{args_str}]"
        return origin_str

    # normal type
    if not hasattr(typ, "__module__"):
        return str(typ)

    module = typ.__module__
    qualname = typ.__qualname__

    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def type_from_string(type_string: str) -> Type:
    """string -> type"""
    # None
    if type_string == "NoneType":
        return type(None)

    # builtin type
    if "." not in type_string and "[" not in type_string:
        import builtins

        return getattr(builtins, type_string)

    # generic type
    if "[" in type_string:
        # TODO(zjx): here we need more complex parsing logic
        pass

    # module type
    module_name, class_name = type_string.rsplit(".", 1)
    module = importlib.import_module(module_name)

    obj = module
    for part in class_name.split("."):
        obj = getattr(obj, part)

    return obj


def apply_patch_to_type(obj: Any, ftype: Type):
    """
    Apply the patch type to the given object, and make the object pass isinstance check.
    """
    if isinstance(obj, ftype):
        return obj
    # monkey patch the type to the given object.
    obj.__original_class__ = obj.__class__
    obj.__class__ = ftype
    return obj


def isinstance_original(obj: Any, typ: Type) -> bool:
    """
    Check if the object is the original type.
    """
    # check if the object has been patched.
    if hasattr(obj, "__original_class__"):
        return obj.__original_class__ == typ

    # for normal type, use isinstance to check.
    return isinstance(obj, typ)
