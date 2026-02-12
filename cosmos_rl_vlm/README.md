# Vision-Language Model Cosmos-RL SFT Training Guide

This folder supports four modes of running vision-language supervised fine-tuning (SFT) in cosmos-rl with different model architectures and data sources. They all share the same launch script `cosmos_vlm_launcher.py`

## Supported Training Modes

| Mode | Model Architecture | Data Loader | Training Folder | Launcher Script |
|------|-------------------|-------------|----------------|----------------|
| 1 | Qwen3-VL | Normal Data Loader | `qwen3_based_training/` | `cosmos_vlm_launcher.py` |
| 2 | Nemotron-VL | Normal Data Loader | `nemotron_based_training/` | `cosmos_vlm_launcher.py` |
| 3 | Qwen3-VL | i4 Data Loader | `qwen3_based_training/` | `cosmos_vlm_launcher.py` |
| 4 | Nemotron-VL | i4 Data Loader | `nemotron_based_training/` | `cosmos_vlm_launcher.py` |


## Directory Structure

```
cosmos_rl_vlm/
├── cosmos_vlm_launcher.py      # Shared launcher can launch four modes with different configurations
├── i4_data_utils.py            # I4 data loader utils
├── qwen3_based_training/
│   ├── launcher.py             # Normal data loader utils and model utils
│   ├── qwen3_based_vlm.toml    # Configuration toml example
│   └── run.sh                  # Example runing commands on lepton
│
└── nemotron_based_training/
    ├── launcher.py             # Normal data loader utils and model utils
    ├── nemotron_based_vlm.toml # Configuration toml example
    └── run.sh                  # Example runing commands on lepton
```

## Configuration

The `[custom]` section in the TOML configuration file allows you to control which data loader to use and specify the data source configuration. Also, `s3_bucket` and `"s3_credentials_path` can be updated in this section to redefine the s3 authorization.

### Custom Configuration Section

Add the following `[custom]` section to your TOML configuration file:

```toml
[custom]
use_i4 = true
i4_data_source = "i4_data_utils.data_weight_default"
s3_bucket = "nv-cosmos-zu-videos"
s3_credentials_path = "credentials/s3_training.secret"
```

The above `i4_data_source`, `s3_bucket` and `"s3_credentials_path` are also the default value if not specified in toml.

#### Configuration Parameters

- **`use_i4`** (boolean):
  - Set to `true` to use the i4 data loader (modes 3 and 4)
  - Set to `false` to use the normal data loader (modes 1 and 2)
  - This flag determines which data loading mechanism is used during training

- **`i4_data_source`** (string):
  - Specifies the path to the data weight configuration for i4 data loader
  - Format: `"module_path.attribute_name"` (e.g., `"i4_data_utils.data_weight_default"`)
  - This should point to a dictionary that defines dataset weights for sampling
  - The module must be importable and contain the specified attribute
  - Example structure in `i4_data_utils.py`:
    ```python
    data_weight_default = {
        "dataset_name_1": 1000000,
        "dataset_name_2": 500000,
        # ... more datasets with their weights
    }
    ```

The `cosmos_vlm_launcher.py` script reads these configuration values to automatically select the appropriate data loader mode and data source configuration for your training job. Also it depends on the specified `model_name_or_path` in the toml configuration to automatically decide training for nemotron-vl or qwen3-vl. Therefore, it can choose among the supported four modes to launch the training with given configuration toml.

