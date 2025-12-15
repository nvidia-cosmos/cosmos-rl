from cosmos_rl.policy.model.base import WeightMapper


class DiffuserModelWeightMapper(WeightMapper):
    def __init__(self, diffusers_config):
        super().__init__(diffusers_config)

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        pass

    def rollout_map_local_key_to_hf_key(self, name: str) -> str:
        pass