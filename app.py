"""
    Modified from: https://github.com/PKU-YuanGroup/ConsisID/blob/main/app.py
"""

import math
import os
import random
import threading
import time
from datetime import datetime, timedelta
from PIL import Image, ImageOps
import numpy as np 

import gradio as gr
from diffusers import CogVideoXDPMScheduler
import spaces
import torch
from models.utils import process_face_embeddings_split
from models.pipeline_ingredients import IngredientsPipeline
from moviepy import VideoFileClip

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory
import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.utils import export_to_video, load_image, load_video
from models.transformer_ingredients import IngredientsTransformer3DModel
from models.eva_clip import create_model_and_transforms
from models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from models.eva_clip.utils_qformer import resize_numpy_image_long

# 1. prepare configs
model_path = "/maindata/data/shared/public/zhengcong.fei/ckpts/cogvideox1.5/consistent_id_sam" 
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

output_path = "output"

# 0. load main models
if os.path.exists(os.path.join(model_path, "transformer_ema")):
    subfolder = "transformer_ema"
else:
    subfolder = "transformer"
    
transformer = IngredientsTransformer3DModel.from_pretrained_cus(model_path, subfolder=subfolder)
transformer.eval()
scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

try:
    is_kps = transformer.config.is_kps
except:
    is_kps = False

print("is kps", is_kps)
# 1. load face helper models
face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    device=device,
    model_rootpath=os.path.join(model_path, "face_encoder")
)
face_helper.face_parse = None
face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=os.path.join(model_path, "face_encoder"))
face_helper.face_det.eval()
face_helper.face_parse.eval()

model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
face_clip_model = model.visual
face_clip_model.eval()

eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
if not isinstance(eva_transform_mean, (list, tuple)):
    eva_transform_mean = (eva_transform_mean,) * 3
if not isinstance(eva_transform_std, (list, tuple)):
    eva_transform_std = (eva_transform_std,) * 3
eva_transform_mean = eva_transform_mean
eva_transform_std = eva_transform_std

face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(model_path, "face_encoder"), providers=['CUDAExecutionProvider'])
handler_ante = insightface.model_zoo.get_model(f'{model_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'])
face_main_model.prepare(ctx_id=0, det_size=(640, 640))
handler_ante.prepare(ctx_id=0)
    
face_clip_model.to(device, dtype=dtype)
face_helper.face_det.to(device)
face_helper.face_parse.to(device)
transformer.to(device, dtype=dtype)
free_memory()

pipe = IngredientsPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)

# 2. Set Scheduler.
scheduler_args = {}
if "variance_type" in pipe.scheduler.config:
    variance_type = pipe.scheduler.config.variance_type
    if variance_type in ["learned", "learned_range"]:
        variance_type = "fixed_small"
    scheduler_args["variance_type"] = variance_type

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

# 3. Enable CPU offload for the model.
pipe.to(device)


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def generate(
    prompt,
    input_image1,
    input_image2,
    negative_prompt: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # turn on if you don't have multiple GPUs or enough GPU memory(such as H100) and it will cost more time in inference, it may also reduce the quality
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    
    # process face data 
    img_file_path_list = [input_image1, input_image2]
        
    print(len(img_file_path_list))
    print(prompt)

    id_cond_list = []
    id_vit_hidden_list = [] 

    id_image_list = []
    for img_file_path in img_file_path_list:
        id_image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))
        id_image = resize_numpy_image_long(id_image, 1024)
        id_image_list.append(id_image)
    id_cond_list, id_vit_hidden_list, align_crop_face_image, face_kps, _ = process_face_embeddings_split(face_helper, face_clip_model, handler_ante, 
                                                                            eva_transform_mean, eva_transform_std, 
                                                                            face_main_model, device, dtype, id_image_list, 
                                                                            original_id_images=id_image_list, is_align_face=True, 
                                                                            cal_uncond=False)
    if is_kps:
        kps_cond = face_kps
    else:
        kps_cond = None
    print("kps_cond: ", kps_cond, "align_face: ", align_crop_face_image.size(), )
    print("id_cond: ", len(id_cond_list), ) 

    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    prompt = prompt.strip('"')
    
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    
    with torch.no_grad(): 
        video_generate = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=generator,
            id_vit_hidden=id_vit_hidden_list,
            id_cond=id_cond_list,
            kps_cond=kps_cond,
        ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    filename = f"{output_path}/results.mp4" 
    print(filename)
    export_to_video(video_generate, filename, fps=8) 
    return filename



with gr.Blocks() as demo:
    gr.Markdown("""
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://raw.githubusercontent.com/feizc/Ingredients/refs/heads/main/asserts/logo.jpg' style='width: 350px; height: auto; margin-right: 10px;' />
        </div>
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            Ingredients Space
        </div>
        <div style="text-align: center;">
            <a href="https://huggingface.co/feizhengcong/Ingredients">🤗 Model Hub</a> |
            <a href="https://huggingface.co/datasets/feizhengcong/Ingredients">📚 Dataset</a> |
            <a href="https://github.com/feizc/Ingredients">🌐 Github</a> |
            <a href="https://arxiv.org/abs/2501.01790">📜 arxiv </a>
        </div>
        <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
        ⚠️ This demo is for academic research and experiential use only. 
        </div>
        """)
    
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Multi-ID Image Input", open=True):
                image_input1 = gr.Image(label="Input Image 1 (should contain clear face, preferably half-body or full-body image)")
                image_input2 = gr.Image(label="Input Image 2 (should contain clear face, preferably half-body or full-body image)")
                prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here. Ingredients has high requirements for prompt quality. You can use GPT-4o to refine the input text prompt, example can be found on our github.", lines=5)
                negative_prompt = gr.Textbox(label="Negative Prompt (Default is None)", placeholder="Enter your negative prompt here. Default is None", lines=1)
        
            with gr.Group():
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=2025
                        )
                        cfg_param = gr.Number(
                            label="Guidance Scale (Enter a positive number, default = 6.0)", value=6.0
                        )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Ingredients Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    def run(
        prompt,
        image_input1,
        image_input2,
        negative_prompt,
        num_inference_steps,
        cfg_param,
        seed_value,
        progress=gr.Progress(track_tqdm=True)
    ):
        video_path = generate(
            prompt,
            image_input1, 
            image_input2, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_param,
            seed=seed_value,
        )

        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)
        seed_update = gr.update(visible=True, value=seed_value)

        return video_path, video_update, gif_update, seed_update

    generate_button.click(
        fn=run,
        inputs=[prompt, image_input1, image_input2, negative_prompt, num_inference_steps, cfg_param, seed_param, ],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )



if __name__ == "__main__":
    demo.queue(max_size=15)
    demo.launch(share=True)