import torch
from typing import Optional
from transformers import AutoConfig


# For transformers v5.0.0 and higher, "default" is excluded from dictionary ROPE_INIT_FUNCTIONS and harcoded individually for each model.
# This function is a copy of the original function in Transformers < 5.0.0, to be compatible with Transformers v5.0.0 and higher.
def _compute_default_rope_parameters(
    config: Optional[AutoConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = getattr(config, "rope_theta", None) or getattr(
        config, "rope_parameters", {}
    ).get("rope_theta", None)
    if base is None:
        raise ValueError("rope_theta is not found in config={config}")
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor
