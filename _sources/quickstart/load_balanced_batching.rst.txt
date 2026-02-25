Load-Balanced Dynamic Batching
================================

Overview
--------

Load-balanced dynamic batching is a data loading strategy designed to minimize padding waste and improve training efficiency in distributed training scenarios. Unlike traditional fixed-size batching, this approach dynamically creates batches that maximize token utilization while respecting a maximum token constraint per batch.

Key Features
------------

- **Dynamic Batch Formation**: Batches are created on-the-fly based on sample lengths, maximizing batch size while staying within token limits
- **Load Balancing**: Balances the number of tokens across different data parallel ranks, reducing padding waste
- **Step-Based Training**: Training is controlled by optimizer steps (``max_num_steps`` in ``[train]``), not epochs. User-provided epoch configuration is ignored
- **Automatic Data Looping**: When ``infinite_loop = true`` (default), data automatically restarts when exhausted, with epoch incremented for new data ordering
- **Gradient Accumulation Support**: Built-in support for accumulating multiple batches per optimizer step
- **Resume Support**: Properly handles training resumption with deterministic data ordering based on train_step

How It Works
------------

The load-balanced batching system consists of two main components:

1. **ShardedIterableDataset**: Shards the base dataset across data parallel ranks and converts it to an IterableDataset
2. **LoadBalancedDataset**: Maintains a pool of samples and dynamically creates batches using a best-fit strategy

**Training Mode**:
- When ``enable_dp_load_balancing = true``, training is **step-based**, not epoch-based
- Training duration is controlled by ``max_num_steps`` in ``[train]`` (number of optimizer steps)
- User-provided ``epoch`` configuration parameter is **ignored**
- Epoch is managed internally for deterministic data ordering (different epoch = different shuffle)
- When ``infinite_loop = true`` (default), data automatically restarts when exhausted, with epoch incremented
- Each rank may consume data at different rates due to dynamic batching, but training stops when ``total_steps`` is reached

Batch Formation Strategy
------------------------

The system uses a pool-based approach:

1. **Sample Pool**: Each rank maintains a pool of samples (default: 32 samples)
2. **Best-Fit Selection**: When forming a batch, the system selects samples from the pool based on the batching mode:
   
   **Without Sequence Packing** (default):
   - Maximizes batch_size * max_input_len while staying within ``max_tokens_for_batch``
   - Uses padding to align sequences to the same length
   - Batching strategies:
     - ``prefer_closest``: Selects samples with lengths closest to existing samples in the batch (minimizes padding)
     - ``prefer_first``: FIFO selection (faster but may have more padding)
   
   **With Sequence Packing** (``sequence_packing = true``):
   - Maximizes total tokens (sum of all sequence lengths) while staying within ``max_tokens_for_batch``
   - Multiple sequences are packed into a single tensor without padding
   - Uses a simpler greedy strategy: adds sequences until total tokens exceed the limit
   - More efficient token utilization, but requires model support for sequence packing

Configuration
-------------

Load-balanced batching is configured through the training policy configuration:

.. code-block:: toml

   [train]
   max_num_steps = 100  # Required when enable_dp_load_balancing is true
   sequence_packing = false  # Set to true to enable sequence packing

   [train.train_policy]
   enable_dp_load_balancing = true
   load_balanced_pool_size = 32
   load_balanced_max_tokens_for_batch = 32768
   load_balanced_batching_strategy = "prefer_closest"  # or "prefer_first"
   load_balanced_batches_per_optimizer_step = 1  # Also known as load_balanced_accumulate_steps

Configuration Parameters
------------------------

enable_dp_load_balancing
   Enable load-balanced dynamic batching (default: false)

load_balanced_pool_size
   Size of the sample pool maintained by each rank (default: 32)
   
   Larger pool sizes allow better batch formation but use more memory.

load_balanced_max_tokens_for_batch
   Maximum number of tokens per batch (default: 32768)
   
   This is the primary constraint for batch formation. The system will create batches that maximize batch_size * max_input_len while staying within this limit.

load_balanced_batching_strategy
   Batching strategy: "prefer_closest" or "prefer_first" (default: "prefer_closest")
   
   - ``prefer_closest``: Minimizes padding by selecting samples with similar lengths
   - ``prefer_first``: FIFO selection, faster but may have more padding

max_num_steps (in ``[train]``)
   Maximum number of optimizer steps (training steps). **Required** when ``enable_dp_load_balancing = true``.
   
   This defines the number of times ``optimizer.step()`` will be called. The actual number of batches processed will be ``max_num_steps * load_balanced_batches_per_optimizer_step``.
   
   **Important**: When ``enable_dp_load_balancing = true``, training is **step-based**, not epoch-based. The user-provided ``epoch`` configuration parameter is ignored. The system uses ``max_num_steps`` (in ``[train]`` section) to determine when training should stop.

load_balanced_batches_per_optimizer_step
   Number of batches to accumulate per optimizer step for gradient accumulation (default: 1)
   
   Each DataLoader iteration will return this many batches, which are processed before calling ``optimizer.step()``. The total number of batches processed = ``max_num_steps`` (in ``[train]``) * ``load_balanced_batches_per_optimizer_step``.

sequence_packing
   Enable sequence packing for training (default: false)
   
   When enabled, multiple sequences are packed into a single tensor without padding, maximizing token utilization. The batch formation strategy changes from maximizing ``batch_size * max_input_len`` to maximizing ``sum(sequence_lengths)`` within the token limit.
   
   **Important**: Sequence packing requires model support. Not all models support sequence packing. The system will check compatibility and warn if the model doesn't support it.
   
   When sequence packing is enabled:
   - The batching strategy (``prefer_closest`` vs ``prefer_first``) is ignored
   - A greedy algorithm is used: sequences are added until total tokens exceed the limit
   - More efficient token utilization compared to padding-based batching
   - Requires the model to handle variable-length sequences within a batch

Usage Example
-------------

Here's a complete configuration example:

.. code-block:: toml

   [train]
   max_num_steps = 1000

   [train.train_policy]
   enable_dp_load_balancing = true
   load_balanced_pool_size = 64
   load_balanced_max_tokens_for_batch = 65536
   load_balanced_batching_strategy = "prefer_closest"
   load_balanced_batches_per_optimizer_step = 4
   dataloader_seed = 42

In this example:
- Each rank maintains a pool of 64 samples
- Maximum tokens per batch is 65536
- Uses "prefer_closest" strategy to minimize padding
- Training will run for 1000 optimizer steps
- Each optimizer step accumulates 4 batches (gradient accumulation)
- Total batches processed = 1000 * 4 = 4000 batches

Example with Sequence Packing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [train]
   sequence_packing = true
   max_num_steps = 1000

   [train.train_policy]
   enable_dp_load_balancing = true
   load_balanced_pool_size = 64
   load_balanced_max_tokens_for_batch = 65536
   load_balanced_batches_per_optimizer_step = 4
   dataloader_seed = 42

In this example with sequence packing:
- Sequence packing is enabled (no padding needed)
- Maximum total tokens per batch is 65536 (sum of all sequence lengths)
- The batching strategy is automatically set to greedy packing
- More efficient token utilization compared to padding-based batching

Gradient Accumulation
---------------------

Load-balanced batching supports gradient accumulation at the data loading level. When ``load_balanced_batches_per_optimizer_step > 1``:

1. Each DataLoader iteration returns a list of batches (instead of a single batch)
2. The trainer processes all batches in the list, accumulating gradients
3. A single ``optimizer.step()`` is called after processing all batches

This approach moves gradient accumulation logic from the trainer to the data loading layer, providing better modularity and efficiency.

Infinite Loop and Epoch Management
----------------------------------

When ``enable_dp_load_balancing = true``, the system uses an ``infinite_loop`` parameter to control data iteration behavior:

**Infinite Loop (default: true)**:
- When ``infinite_loop = true``: Data automatically restarts when exhausted
  - Epoch is automatically incremented each time data restarts
  - Different epochs use different random seeds for data shuffling (deterministic but varied ordering)
  - This ensures training can reach ``max_num_steps`` even if data is exhausted
  - Recommended for step-based training where you want to train for a fixed number of optimizer steps

- When ``infinite_loop = false``: Data stops when exhausted
  - Training stops when all data has been processed
  - Not recommended for step-based training as training may stop before reaching ``max_num_steps``

**Epoch Management**:
- Epoch is **managed internally** and is used only for deterministic data ordering
- User-provided ``epoch`` configuration parameter is **ignored** when ``enable_dp_load_balancing = true``
- Epoch does not control training duration (training is step-based, controlled by ``max_num_steps``)
- When data restarts (``infinite_loop = true``), epoch is automatically incremented to ensure different data ordering
- Initial epoch is set to 0 when resuming training

**Why Infinite Loop?**:
- In dynamic batching, different ranks consume data at different rates
- Some ranks may exhaust their data shard before reaching ``max_num_steps``
- With ``infinite_loop = true``, data automatically restarts, ensuring all ranks can continue training
- Training stops when ``train_step >= total_steps`` (where ``total_steps = max_num_steps``), not when data is exhausted

Resume Support
--------------

Load-balanced batching properly handles training resumption:

1. **Step-Based Training**: Training is based on optimizer steps (``max_num_steps``), not epochs
2. **Automatic Epoch Management**: Epoch is automatically managed internally for deterministic data ordering
   - Initial epoch is set to 0 when resuming
   - Epoch is automatically incremented when data loops (if ``infinite_loop = true``)
   - Different epochs use different random seeds for data shuffling
3. **Batch Skipping**: Skips batch groups that have already been processed based on ``train_step``

The resume logic ensures that:
- Data ordering matches the original training (deterministic shuffling)
- Only batches within the current step range are skipped
- Training continues from the correct position

**Note**: The user-provided ``epoch`` configuration parameter is **ignored** when ``enable_dp_load_balancing = true``. Epoch is managed internally for data ordering purposes only.

Implementation Details
----------------------

ShardedIterableDataset
~~~~~~~~~~~~~~~~~~~~~~

The ``ShardedIterableDataset`` class:

- Shards the base dataset across data parallel ranks
- Converts a regular ``Dataset`` to an ``IterableDataset``
- Supports deterministic shuffling based on epoch number
- Ensures each rank only sees its portion of the data

LoadBalancedDataset
~~~~~~~~~~~~~~~~~~

The ``LoadBalancedDataset`` class:

- Maintains a pool of samples for dynamic batch formation
- Implements best-fit batching strategies (with or without sequence packing)
- Supports gradient accumulation by yielding multiple batches per iteration
- Provides ``set_epoch()`` and ``skip_batches()`` methods for resume support
- Supports automatic data looping with ``infinite_loop`` parameter:
  - When ``infinite_loop = true`` (default): Automatically restarts data iteration when exhausted, incrementing epoch for new data ordering
  - When ``infinite_loop = false``: Stops iteration when data is exhausted
- Adapts batch formation algorithm based on ``seq_packing_enabled`` flag:
  - Without packing: maximizes batch_size * max_length (with padding)
  - With packing: maximizes sum of sequence lengths (without padding)

**Epoch Management**:
- Epoch is managed internally for deterministic data ordering (different epoch = different shuffle)
- When ``infinite_loop = true``, epoch is automatically incremented each time data restarts
- Epoch does not control training duration (training is step-based, controlled by ``max_num_steps``)

Best-Fit Algorithm
~~~~~~~~~~~~~~~~~~

The best-fit algorithm works differently depending on whether sequence packing is enabled:

**Without Sequence Packing** (default):

1. Start with an empty batch
2. For each sample in the pool:
   - Calculate the new batch size and max length if this sample is added
   - Check if the total tokens (batch_size * max_length) <= max_tokens_for_batch
   - If valid, calculate a score based on the batching strategy
3. Select the sample with the best score (highest for "prefer_closest", first for "prefer_first")
4. Add the sample to the batch and remove it from the pool
5. Repeat until no more samples can be added

**With Sequence Packing** (``sequence_packing = true``):

1. Start with an empty batch
2. For each sample in the pool (in order):
   - Calculate the new total tokens (sum of all sequence lengths) if this sample is added
   - Check if the total tokens <= max_tokens_for_batch
   - If valid, add the sample to the batch and remove it from the pool
3. Repeat until no more samples can be added

The sequence packing algorithm uses a greedy approach: it adds sequences until the total token count exceeds the limit, maximizing token utilization without padding.

Advantages
----------

1. **Reduced Padding Waste**: By grouping samples with similar lengths, padding is minimized. With sequence packing enabled, padding is completely eliminated
2. **Better GPU Utilization**: More tokens per batch means better GPU utilization
3. **Flexible Batch Sizes**: Adapts to varying sample lengths automatically
4. **Distributed Training Friendly**: Balances load across ranks while maintaining data distribution
5. **Sequence Packing Support**: When enabled, eliminates padding entirely by packing multiple sequences into a single tensor, maximizing token efficiency

Limitations
-----------

1. **Approximate Length**: ``len(dataset)`` returns an approximate value based on sample count, not actual batch count
2. **Memory Overhead**: Maintaining a sample pool requires additional memory
3. **Step-Based Training**: When ``enable_dp_load_balancing = true``, training is step-based, not epoch-based. User-provided epoch configuration is ignored
4. **Epoch Management**: Epoch is managed internally for data ordering purposes only. It does not control training duration
5. **Deterministic Resume**: Resume is deterministic based on train_step, but exact batch composition may vary slightly due to dynamic batching

Best Practices
--------------

1. **Pool Size**: Choose a pool size that balances memory usage and batch quality. Larger pools (64-128) work well for most cases
2. **Max Tokens**: Set ``max_tokens_for_batch`` based on your GPU memory and model size. Common values: 32768, 65536, 131072
3. **Batching Strategy**: Use "prefer_closest" for better efficiency, "prefer_first" if speed is more important (only applies when sequence packing is disabled)
4. **Sequence Packing**: Enable sequence packing if your model supports it for better token utilization. Check model compatibility before enabling
5. **Gradient Accumulation**: Use ``load_balanced_batches_per_optimizer_step`` to control effective batch size
6. **Seed**: Set ``dataloader_seed`` for reproducibility
7. **Step-Based Training**: Remember that when ``enable_dp_load_balancing = true``, training is step-based. Set ``max_num_steps`` (in ``[train]``) to control training duration, not ``epoch``
8. **Infinite Loop**: Keep ``infinite_loop = true`` (default) for step-based training. Data will automatically restart when exhausted, ensuring training can reach ``max_num_steps``

When to Use Sequence Packing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence packing is recommended when:
- Your model supports variable-length sequences within a batch
- You want to maximize token utilization (reduce padding waste)
- You have sequences with highly variable lengths
- Your model architecture can handle packed sequences efficiently

Sequence packing is NOT recommended when:
- Your model doesn't support sequence packing (check compatibility)
- You need fixed-size batches for certain operations
- The overhead of handling variable-length sequences outweighs the benefits

Troubleshooting
---------------

**Issue**: Training seems slower than expected
   - Check if ``load_balanced_pool_size`` is too small
   - Verify ``max_tokens_for_batch`` is appropriate for your hardware
   - Consider using "prefer_first" strategy for faster batch formation

**Issue**: Out of memory errors
   - Reduce ``load_balanced_max_tokens_for_batch``
   - Reduce ``load_balanced_pool_size``
   - Reduce ``load_balanced_batches_per_optimizer_step``

**Issue**: Resume doesn't work correctly
   - Ensure ``max_num_steps`` (in ``[train]``) matches the original training configuration
   - Check that checkpoint contains correct ``train_step`` information
   - Verify ``dataloader_seed`` is the same as original training
   - Note: User-provided ``epoch`` parameter is ignored when ``enable_dp_load_balancing = true``. Epoch is managed internally

**Issue**: Data keeps looping indefinitely
   - This is expected behavior when ``infinite_loop = true`` (default)
   - Training stops based on ``max_num_steps``, not data exhaustion
   - Set ``infinite_loop = false`` if you want data to stop when exhausted (not recommended for step-based training)

**Issue**: Sequence packing not working
   - Verify that ``sequence_packing = true`` is set in the configuration
   - Check if your model supports sequence packing (the system will warn if not)
   - Ensure ``enable_dp_load_balancing = true`` is also set
   - Check model compatibility: not all models support sequence packing

Related Documentation
---------------------

- :doc:`configuration` - General configuration guide
- :doc:`dataflow` - Data flow in cosmos-rl
