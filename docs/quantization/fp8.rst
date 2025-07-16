FP8 Quantization
================

Cosmos-RL supports FP8 quantization for policy and rollout. Currently, only rowwise quantization is supported.


Enable FP8 Quantization
-----------------------

To enable FP8 quantization, you need to set the following configuration:

.. code-block:: toml

    [train.fp8]
    enable_fp8 = true
    fp8_recipe = "dynamic_scaling"
    quant_recipe = "rowwise"

    [rollout]
    quantization = "fp8"


For policy:

- `fp8_recipe` could only be set to `dynamic_scaling` now. 
- `quant_recipe` could be set to `rowwise` or `tensorwise`. `rowwise` is recommended for better accuracy.

For rollout:

- `quantization` should be set to `fp8`. Then the rollout will dynamically quantize weights from policy to `fp8` during weight synchronization in `rowwise` manner.






