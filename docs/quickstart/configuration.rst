Configuration
================================

.. jsonschema:: config.COSMOS_CONFIG_SCHEMA
    :lift_title: false

Trajectory payload normalization
-------------------------------

NCCL and UCXX trajectory producers share the same schema normalization rules:

* Present fields are cast to the declared wire dtype. Scalar shapes, singleton
  strides, non-contiguous tensors and reversed NumPy views are supported. BF16
  source tensors can be converted to supported wire types such as float32.
* Fixed-size fields must match their declared shape. Sequence fields may be
  shorter only in their leading dimension; trailing dimensions must match.
  They are zero-padded without broadcasting. Absent optional fields are zero.
* ``episode_length`` is a nonnegative integer, encoded as an int64 scalar or
  one-element vector. It cannot exceed the schema capacity or the available
  rows of a supplied sequence field. Consumers validate it before truncation.

Invalid payloads are rejected through the existing producer failure/fallback
path. A failed shared-memory write does not leave a slot permanently claimed.
These rules do not alter training objectives or transport recovery policy.
Packed policy-to-policy state and rollout-to-rollout weight synchronization
also share the byte-layout helpers, including copy-back into singleton-strided
destination views. Their opt-in flags and transfer schedules are unchanged.

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
