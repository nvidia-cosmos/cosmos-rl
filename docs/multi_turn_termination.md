# Multi-turn termination and limits

The synchronous vLLM adapter stops when `extend_conversation` leaves a final
assistant answer, or adds no messages. A packer that wants another model turn
must append a tool observation or user message. The base packer appends only the
assistant response; the bundled GSM8K packer appends tool observations only when
a tool was actually called.

`max_assistant_turns` must be positive and bounds generated assistant turns per
sample. The selected training/validation sampling parameters continue to cap
each assistant response. Prompt/history tokens do not consume that response
cap; the separate `policy.model_max_length` bounds the model context. The adapter
stops when the returned prompt plus response reaches that context limit. vLLM's
input validation still applies if a tool observation makes the next input too
large; this change does not silently truncate conversations or tool outputs.

When a turn/context limit is reached after a tool call, the reward completion
is the last generated assistant response, not the appended tool observation.
Each sampled trajectory keeps its own conversation copy. This does not change
single-turn generation, OpenVLA/PI05 objectives, multi-turn loss formulation or
the existing prompt-logprob representation. Multi-turn distillation with
different sampled histories is not validated by these tests.

`tests/test_multi_turn_termination.py` runs the actual adapter method with
controlled engine outputs. `tests/multi_turn_termination_canary.py` uses actual
vLLM generation and dummy weights, with a deterministic packer controlling tool
continuation. It checks final answers, tool continuation, turn-limited reward
selection, two independent samples and different training/validation response
caps. It is not a model-accuracy, real-tool or full-controller test.
