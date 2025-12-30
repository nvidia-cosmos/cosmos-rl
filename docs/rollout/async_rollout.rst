\[Experimental\] Asynchronous Rollout
=============================

Cosmos-RL supports asynchronous rollout generation.


Enable Asynchronous Rollout
-------------------

To enable asynchronous rollout, you need to set the fields of ``rollout`` section in the config file:

.. code-block:: toml

    [rollout]
    backend = "vllm_async"
    mode = "async"
    async_config.max_concurrent_requests = 20


For now, tested backend are:
- vllm_async

