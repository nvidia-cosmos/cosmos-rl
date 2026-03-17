Overview
=================

Cosmos-RL provides native support for SFT and RL of world foundational models.

Supported Models
----------------

Cosmos-RL uses diffusers-based training pipelines for both WFM SFT and WFM RL, so the same overall workflow applies across image and video diffusion models. The exported checkpoints are also diffusers-compatible, which makes post-training inference straightforward.

- SD3
- Cosmos-Predict2.5
- SANA-Image/Video

Cosmos-RL supports both LoRA finetuning and full-model finetuning.


Configurations
--------------


The full configuration schema is defined in `Configuration <https://nvidia-cosmos.github.io/cosmos-rl/quickstart/configuration.html>`_. In practice, the most important config groups for WFM jobs are:

- ``[policy]``: sets ``model_name_or_path``, enables diffusers with ``is_diffusers = true``, and optionally enables LoRA through ``policy.lora``.
- ``[policy.diffusers]``: controls model-specific behavior such as ``is_video``, ``max_prompt_length``, ``inference_size``, ``train_frames``, and the sampling block under ``policy.diffusers.sample``.
- ``[policy.diffusers.sample]``: controls rollout and inference behavior such as ``num_steps``, ``eval_num_steps``, ``guidance_scale``, ``noise_level``, ``solver``, and ``deterministic_sampling``.
- ``[policy.parallelism]``: defines ``cp_size``, and ``dp_shard_size`` for the trainer-side mesh.
- ``[train]``: controls optimization and runtime, especially ``output_dir``, ``param_dtype``, ``fsdp_reduce_dtype``, ``train_batch_per_replica``, ``ema_enable``, and ``compile``.
- ``[train.ckpt]``: controls checkpoint cadence and diffusers export. Keep ``export_safetensors = true`` if you want ready-to-load diffusers checkpoints under ``train.output_dir/safetensors/``.
- ``[validation]``: configures periodic evaluation data and validation frequency.

SFT
^^^

Starter SFT recipes are available under ``configs/stable-diffusion-3-5/`` and ``configs/sana/``:

- SD3: `stable-diffusion-3-5-image-sft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/stable-diffusion-3-5/stable-diffusion-3-5-image-sft.toml>`_, `stable-diffusion-3-5-image-sft-lora.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/stable-diffusion-3-5/stable-diffusion-3-5-image-sft-lora.toml>`_
- SANA image: `sana-image-sft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-sft.toml>`_, `sana-image-sft-lora.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-sft-lora.toml>`_
- SANA video: `sana-video-sft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-sft.toml>`_, `sana-video-sft-lora.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-sft-lora.toml>`_

The most important SFT-specific settings are:

- Set ``train.train_policy.type = "sft"``.
- Point ``train.train_policy.dataset.local_dir`` at your local image or video dataset.
- Add ``[policy.lora]`` for adapter finetuning. If you omit it, the trainer updates the full transformer.

RL
^^

Starter RL recipes are also available under ``configs/stable-diffusion-3-5/`` and ``configs/sana/``:

- SD3: `stable-diffusion-3-5-medium-nft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/stable-diffusion-3-5/stable-diffusion-3-5-medium-nft.toml>`_
- SANA image: `sana-image-nft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-image-nft.toml>`_
- SANA video: `sana-video-nft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/sana/sana-video-nft.toml>`_
- Cosmos-Predict2.5: `cosmos-predict2-5-2b-720-nft.toml <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/cosmos-predict2-5/cosmos-predict2-5-2b-720-nft.toml>`_

The most important RL-specific settings are:

- Set ``train.train_policy.type = "grpo"`` and ``train.train_policy.trainer_type = "diffusion_nft"``.
- Configure prompt sampling and rollout behavior through ``[rollout]`` and ``[policy.diffusers.sample]``.
- Enable remote rewards with ``train.train_policy.use_remote_reward = true`` and define one or more ``[[train.train_policy.remote_reward.reward_fns]]`` entries.
- Keep ``rollout.parallelism`` and ``policy.parallelism`` aligned in colocated mode, especially ``dp_shard_size``.


Datasets
--------

The dataset entrypoint is a Python launcher, so the main customization path is to edit the corresponding dataset file rather than changing a fixed built-in schema. For more details about dataset customization, please refer to `Customization <https://nvidia-cosmos.github.io/cosmos-rl/quickstart/customization.html>`_.

SFT
^^^

SFT uses ``cosmos_rl.tools.dataset.diffusers_dataset``.

- Point ``train.train_policy.dataset.local_dir`` to a local directory of paired metadata and visual assets.
- Each sample is expected to share the same basename across metadata and asset files, for example ``0001.json`` with ``0001.jpg`` for images, or ``0001.json`` with ``0001.mp4`` for videos.
- ``policy.diffusers.is_video`` and ``policy.diffusers.train_frames`` determine whether the loader expects images or videos and how many frames are sampled for video training.

RL
^^

RL uses ``cosmos_rl.tools.dataset.diffusion_nft``.

- The built-in launcher currently supports prompt datasets such as ``pickscore``, ``ocr``, ``geneval``, and ``dance_grpo_t2v`` through ``train.train_policy.dataset.name`` and ``split``.
- These datasets are prompt-first rather than paired image/video supervision datasets; the reward signal is provided separately by the reward service.
- For custom prompt sources, metadata packing, or reward-service payloads, edit ``cosmos_rl.tools.dataset.diffusion_nft.py``.

Launch Training
---------------

Install the WFM dependencies first::

    pip install '.[wfm]'

If you are training Cosmos-Predict2.5, also install::

    pip install cosmos_guardrail --no-deps

Training progress can be monitored in Weights & Biases when ``logging.logger`` includes ``wandb``.

SFT
^^^

Launch SFT with ``cosmos_rl.tools.dataset.diffusers_dataset``. For example, SD3 LoRA SFT::

    cosmos-rl --config ./configs/stable-diffusion-3-5/stable-diffusion-3-5-image-sft-lora.toml cosmos_rl.tools.dataset.diffusers_dataset

To run full-model SFT, switch to ``stable-diffusion-3-5-image-sft.toml`` or one of the SANA ``*-sft.toml`` recipes.

RL
^^

Launch a reward service first by following `Reward Service <https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service>`_. Then point the trainer to that service through environment variables::

    export REMOTE_REWARD_TOKEN="your_token"
    export REMOTE_REWARD_ENQUEUE_URL="https://reward-service-host:PORT/api/reward/enqueue"
    export REMOTE_REWARD_FETCH_URL="https://reward-service-host:PORT/api/reward/fetch"

After that, launch DiffusionNFT training with ``cosmos_rl.tools.dataset.diffusion_nft``. For example, SD3 RL::

    cosmos-rl --config ./configs/stable-diffusion-3-5/stable-diffusion-3-5-medium-nft.toml cosmos_rl.tools.dataset.diffusion_nft

To run SANA RL, switch to ``sana-image-nft.toml`` or ``sana-video-nft.toml``.

Use Trained Models
------------------

When ``train.ckpt.export_safetensors = true``, diffusers-compatible artifacts are exported under ``train.output_dir/safetensors/step_<N>/``.
If its false, we will only export the final checkpoint to diffusers-compatible safetensors.

- If ``policy.lora`` is configured, the adapter is saved under ``step_<N>/lora/``.
- If ``policy.lora`` is not configured, the full diffusers pipeline is saved directly under ``step_<N>/``.

LoRA
^^^^

- SFT: load the base pipeline from ``policy.model_name_or_path``, then attach the adapter from ``.../safetensors/step_<N>/lora``.
- RL: the loading flow is the same. RL LoRA checkpoints can be used with regular diffusers inference; the reward service is only needed during training.

.. code-block:: python

    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video

    base_model = "stabilityai/stable-diffusion-3.5-medium"
    adapter_dir = "./outputs/stable-diffusion-3-5-image-sft-lora/safetensors/step_30/lora"

    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(
        adapter_dir,
        weight_name="model.safetensors",
        adapter_name="default",
    )
    pipe.set_adapters("default")
    pipe = pipe.to("cuda")

    result = pipe(
        prompt="A cinematic photo of a corgi astronaut on the moon.",
        num_inference_steps=40,
        guidance_scale=4.5,
    )

    if hasattr(result, "images"):
        result.images[0].save("sample.png")
    else:
        export_to_video(result.frames[0], "sample.mp4", fps=16)

Full Pipeline
^^^^^^^^^^^^^

- SFT: load the exported step directory directly with ``DiffusionPipeline.from_pretrained``.
- RL: the same loading path applies when training without ``policy.lora``.

.. code-block:: python

    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video

    checkpoint_dir = "./outputs/stable-diffusion-3-5-image-sft/safetensors/step_30"

    pipe = DiffusionPipeline.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    result = pipe(
        prompt="A cinematic photo of a corgi astronaut on the moon.",
        num_inference_steps=40,
        guidance_scale=4.5,
    )

    if hasattr(result, "images"):
        result.images[0].save("sample.png")
    else:
        export_to_video(result.frames[0], "sample.mp4", fps=16)


Add New Models
--------------

To add support for a new diffusers-based WFM, the best references are the existing implementations in ``cosmos_rl/policy/model/diffusers/sd3_model.py``, ``cosmos_rl/policy/model/diffusers/sana_model.py``, and ``cosmos_rl/policy/model/diffusers/cosmos_predict2_5_model.py``.

The integration pattern is:

1. Add a new file under ``cosmos_rl/policy/model/diffusers/``.
2. Define a subclass of ``DiffuserModel`` and register it with ``@ModelRegistry.register(DiffuserModelWeightMapper)``.
3. Return the diffusers pipeline class name from ``supported_model_types()``.

The ``supported_model_types()`` value must match ``DiffusionPipeline.load_config(model_name_or_path)["_class_name"]``, because ``ModelRegistry.build_diffusers_model()`` uses that value to choose the implementation class.

New files under ``cosmos_rl/policy/model/diffusers/`` are auto-discovered by ``cosmos_rl.policy.model.__init__``, so placing the new model file in that directory is enough as long as it is a regular ``.py`` module.

.. code-block:: python

    from cosmos_rl.policy.model.base import ModelRegistry
    from cosmos_rl.policy.model.diffusers import DiffuserModel
    from cosmos_rl.policy.model.diffusers.weight_mapper import DiffuserModelWeightMapper
    from cosmos_rl.policy.config import DiffusersConfig


    @ModelRegistry.register(DiffuserModelWeightMapper)
    class MyModel(DiffuserModel):
        @staticmethod
        def supported_model_types():
            return ["MyPipeline"]

        def __init__(self, config: DiffusersConfig, **kwargs):
            super().__init__(config, **kwargs)
            self.set_scheduler_timestep(self.train_sampling_steps)
            self.text_encoders = [self.text_encoder]
            self.tokenizers = [self.tokenizer]

The loaded pipeline is expected to expose a few standard components:

- ``pipeline.transformer``: this is the trainable denoiser used by SFT, RL, EMA, and LoRA code paths.
- ``pipeline.vae`` and ``pipeline.scheduler``: used by latent encoding, decoding, and noise scheduling.
- ``pipeline.image_processor`` or ``pipeline.video_processor``: required by ``DiffuserModel.init_output_process()``.

If an upstream diffusers pipeline does not expose its denoiser as ``transformer`` and instead uses a different attribute such as ``unet``, you should add a thin compatibility layer before trying to reuse the existing trainer stack. The current WFM diffusers path assumes the trainable module is available through ``self.transformer``.

SFT
^^^

For SFT, the minimum required methods are:

- ``text_embedding()``: call ``pipeline.encode_prompt()`` and return a dictionary that matches the keyword arguments expected by ``self.transformer(...)``. For example, SD3 uses ``encoder_hidden_states`` and ``pooled_projections``, while SANA and Cosmos-Predict2.5 do not use pooled projections.
- ``visual_embedding()``: encode input images or videos into training latents. This is usually model-specific because the VAE normalization can differ across pipelines.
- ``set_scheduler_timestep()``: prepare the scheduler and cache the timestep map used by training.
- ``add_noise()``: convert sampled timestep indices into actual scheduler timesteps and return ``(noised_latent, noise, timesteps)``.

The existing models illustrate the main latent-conversion differences:

- SD3 and Cosmos-Predict2.5 use VAE ``shift_factor`` and ``scaling_factor`` when converting pixels to latents.
- SANA image uses ``scaling_factor`` only.
- SANA video applies an extra normalization step with ``latents_mean`` and ``latents_std``.

Once these methods are implemented correctly, ``DiffuserModel.training_sft_step()`` can usually be reused without trainer changes.

RL
^^

To support DiffusionNFT RL, you need two more model-specific entry points:

- ``pipeline_with_logprob()``: performs rollout-time sampling and returns the generated visuals together with ``all_latents`` and ``all_log_probs``.
- ``nft_prepare_transformer_input()``: prepares the keyword arguments for ``self.transformer(...)`` during the RL training step. ``diffusers_trainer/nft_trainer.py`` calls this method directly.

There are two common implementation styles in the current codebase:

- SD3 uses a relatively thin wrapper around diffusers prompt encoding and latent preparation, then delegates the denoising loop to the shared ``run_sampling()`` helper.
- SANA and Cosmos-Predict2.5 implement a custom ``sde_step_with_logprob()`` helper because their transition and log-probability computation is more model-specific.

Keep model-specific logic close to the model implementation rather than the trainer. Examples include:

- classifier-free guidance details
- prompt enhancement logic
- resolution binning
- video-specific conditioning inputs
- custom latent packing or padding rules

Configuration
^^^^^^^^^^^^^

After the model class is added, the config side is usually straightforward:

- Set ``policy.is_diffusers = true``.
- Set ``policy.model_name_or_path`` to the new diffusers repo or local exported pipeline.
- Set ``policy.diffusers.is_video`` correctly so dataset preprocessing and inference use the right visual path.
- Tune ``policy.diffusers.sample`` for the new scheduler and guidance behavior.
- If you want LoRA training, set ``policy.lora.target_modules`` according to the module names under ``self.transformer``.

.. note::
    The current ``DiffuserModel`` implementation only supports ``tp_size = 1`` and ``cp_size = 1``. If a new diffusers backend needs tensor or context parallelism, that support has to be added explicitly.

Validation Checklist
^^^^^^^^^^^^^^^^^^^^

Before wiring a new model into large training runs, it is worth validating these points with a tiny config:

1. Check that ``DiffusionPipeline.load_config(model_name_or_path)["_class_name"]`` matches the string returned by ``supported_model_types()``.
2. Build the model once and confirm that ``self.transformer``, ``self.vae``, ``self.scheduler``, and the expected text encoders/tokenizers are registered.
3. Run ``text_embedding()`` and ``visual_embedding()`` on one small batch and verify the output shapes.
4. Run one SFT forward pass and confirm the returned transformer inputs match the model's forward signature.
5. If RL is needed, run one ``pipeline_with_logprob()`` call and one ``nft_prepare_transformer_input()`` call before launching a full RL job.

RL (deprecated)
----------------

Cosmos-RL supports `FlowGRPO <https://arxiv.org/pdf/2505.05470>`_ and `DDRL <https://arxiv.org/pdf/2512.04332>`_ algorithms for world foundational model reinforcement learning.

**Quick start**: A quick start guide for world foundational model's RL:

1. Configure the training recipe by editing toml files under ``configs/cosmos-predict2-5/``.

2. Launch the reward service, you can refer docs here: `Reward Service <https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service>`_.

3. Launch the training script with the configured recipe::

      cosmos-rl --config ./configs/cosmos-predict2-5/cosmos-predict2-5-2b-480-grpo-mock-data.toml --wfm-mode cosmos_rl.tools.dataset.wfm_rl

4. Monitor training progress via Wandb.

5. Evaluate the trained world foundational model using the evaluation script.
   For Cosmos-Predict2.5, you can refer this repo: `cosmos-predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5>`_.

.. note::
    1. You can find detailed tutorials for DDRL here: `DDRL Tutorials <https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md>`_.
    2. For a quick rollout of the training pipeline, we recommend you use the mock_data config file, i.e., ./configs/cosmos-predict2-5/cosmos-predict2-5-2b-480-grpo-mock-data.toml

**Reward services**: Considering the computation overhead, it's necessary to use a seperated async service for reward computing.

- You can launch a reward service by following the instructions here: `Reward Service <https://github.com/nvidia-cosmos/cosmos-rl/tree/main/reward_service>`_.

- Configure the environment variable ``REMOTE_REWARD_TOKEN``, ``REMOTE_REWARD_ENQUEUE_URL``, and ``REMOTE_REWARD_FETCH_URL`` to make the trainer communicate with the reward service::

    export REMOTE_REWARD_TOKEN="your_token"
    export REMOTE_REWARD_ENQUEUE_URL="https://reward_service_host:PORT/api/reward/enqueue"
    export REMOTE_REWARD_FETCH_URL="https://reward_service_host:PORT/api/reward/fetch"

**Models**:

- Cosmos-Predict2.5-2B/14B

**Parallelism**: Support HSDP/FSDP, and context parallel (CP) for world foundational model training. You can edit the related configurations in the toml file to enable these parallelism techniques.::

    [model]
    fsdp_shard_size = 8
    dp_replicate_size = 4

    [model_parallel]
    context_parallel_size = 2

**Datasets**:

- Local dataset: you can use local dataset for training. We follows the local dataset structure as `Cosmos-Predict2.5 <https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_cosmos_nemo_assets.md>`_. The dataset folder format should be::

    datasets/<your_local_dataset>/
    ├── metas/
    │   └── *.txt
    ├── videos/
    │   └── *.mp4
    └── text_embedding <optional> /
        └── *.pickle

- Webdataset: you need to configure the s3 access via environment variables, then you can use webdataset for training.

    - PROD_S3_CHECKPOINT_ACCESS_KEY_ID: Your S3 access key ID.

    - PROD_S3_CHECKPOINT_SECRET_ACCESS_KEY: Your S3 secret access key.

    - PROD_S3_CHECKPOINT_ENDPOINT_URL: Your S3 endpoint url.

    - PROD_S3_CHECKPOINT_REGION_NAME: Your S3 region name.

**Storage**:

- Local storage: you can use local disk for storing checkpoints and logs.

- S3 storage: you need to configure the s3 access via environment variables, then you can use s3 storage for storing checkpoints and logs.
