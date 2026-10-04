"""Rank-local queue slots with metadata from the original global prompt batch."""


class PromptBatch(list):
    def __init__(self, payloads, required_weight_version):
        super().__init__(payloads)
        self.required_weight_version = required_weight_version


def required_weight_version(payloads):
    if isinstance(payloads, PromptBatch):
        return payloads.required_weight_version
    if not payloads:
        raise ValueError("Empty rollout slot has no global batch metadata")
    return max(payload.weight_version for payload in payloads)
