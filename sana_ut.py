from cosmos_rl.policy.config import Config as CosmosConfig
import toml
from cosmos_rl.policy.model.base import ModelRegistry
from diffusers.utils import export_to_video

toml_file = "sana-video-sft.toml"
with open(toml_file, "r") as f:
    config_dict = toml.load(f)

# Ensure CosmosConfig is available (it's imported at the top now)
# from cosmos_rl.policy.config import Config as CosmosConfig
# Need SFTDataConfig and GrpoConfig for from_dict

loaded_config = CosmosConfig.from_dict(config_dict)
print(loaded_config)
model = ModelRegistry.build_model(loaded_config)

bsz = 1
prompt_list = ["Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional." for _ in range(bsz)]

print(model.current_device())

visual_output = model.inference(
    prompt_list=prompt_list,
    height=480,
    width=832,
    frames=81,
    guidance_scale=4.5,
    inference_step=20,
)

for idx, video in enumerate(visual_output):
    export_to_video(video, f"sana_video_{idx}.mp4", fps=16)
