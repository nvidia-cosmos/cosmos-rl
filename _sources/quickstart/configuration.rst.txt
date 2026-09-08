Configuration
================================

.. jsonschema:: config.COSMOS_CONFIG_SCHEMA
    :lift_title: false

Policy-to-policy state packing
--------------------------------

Policy initialization and dynamic scaling can pack small state tensors into
byte-bounded NCCL transfers. Packing is disabled by default because it requires
an additional device-side copy and scratch buffer. Enable it explicitly in the
training configuration:

.. code-block:: toml

    [train]
    p2p_sync_pack_tensors = true

This setting affects policy-to-policy state synchronization only. It does not
change policy-to-rollout or rollout-to-rollout weight synchronization.
