import os
from setuptools import Distribution
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


TOP_LEVEL_DIR = os.getcwd()


def get_extra_compile_args():
    """Function to get extra compile arguments.

    Returns:
        extra_compile_args (dict): Dictionary of compile flags.
    """
    extra_compile_args = {"cxx": []}
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-DCUTE_ARCH_MMA_SM90A_ENABLED=1",  # Enable SM90A architecture support
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    return extra_compile_args


def make_cuda_ext(
    name, module, sources, include_dirs=None, define_macros=None, extra_flags=None
):
    """Build cuda extensions for custom ops.

    Args:
        name (str): Name of the op.
        module (str): Name of the module with the op.
        source (list): List of source files.
        extra_flags (dict): Any extra compile flags.

    Returns
        cuda_ext (torch.utils.cpp_extension.CUDAExtension): Cuda extension for wheeling.
    """
    kwargs = {"extra_compile_args": extra_flags}
    if include_dirs:
        kwargs["include_dirs"] = [
            os.path.join(os.path.relpath(TOP_LEVEL_DIR), *module.split("."), dir)
            for dir in include_dirs
        ]
    if define_macros:
        kwargs["define_macros"] = define_macros

    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[
            os.path.join(os.path.relpath(TOP_LEVEL_DIR), *module.split("."), src)
            for src in sources
        ],
        **kwargs,
    )


EXT_MODULES = [
    make_cuda_ext(
        name="MultiScaleDeformableAttention",
        module="cosmos_rl.policy.model.c_radio_v3.model.ops",
        sources=[
            "src/ms_deform_attn_cpu.cpp",
            "src/ms_deform_attn_api.cpp",
            "src/ms_deform_attn_cuda.cu",
        ],
        include_dirs=["src"],
        define_macros=[("WITH_CUDA", None)],
        extra_flags=get_extra_compile_args(),
    ),
]


def build_extensions():
    dist = Distribution(
        {"ext_modules": EXT_MODULES, "cmdclass": {"build_ext": BuildExtension}}
    )
    cmd = BuildExtension(dist)
    cmd.build_temp = "build"  # build temp directory
    cmd.build_lib = "."  # where to place built .so files
    dist.ext_modules = EXT_MODULES
    cmd.ensure_finalized()
    cmd.run()
    print("Build complete!")


if __name__ == "__main__":
    build_extensions()
