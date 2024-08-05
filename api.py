#pip install diffusers --upgrade
#pip install invisible_watermark transformers accelerate safetensors

from flask import Flask, request, jsonify, send_file
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os

app = Flask(__name__)

device = "cuda"
dtype = torch.float16

step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose your favorite base model.

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

@app.route('/animateDiff', methods=['POST'])
def generate_animation():
    data = request.json
    prompt = data.get("prompt", "Elon Musk Dancing")
    guidance_scale = data.get("guidance_scale", 1.0)
    num_inference_steps = data.get("num_inference_steps", step)

    output = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    output_path = "animation.gif"
    export_to_gif(output.frames[0], output_path)

    return send_file(output_path, mimetype='image/gif')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9200)

