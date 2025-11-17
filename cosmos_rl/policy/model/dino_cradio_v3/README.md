## Compile the custom kernels
DeformableAttention module relies on custom CUDA kernels for both increased performance and reduced memory usage. To compile the kernels, run the following command:

```
python model/deformable_detr/model/ops/setup.py
```