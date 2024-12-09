import argparse
import logging
import os

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from tqdm import tqdm

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video


logger = logging.getLogger("memo")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for MEMO")

    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--input_image", type=str)
    parser.add_argument("--input_audio", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_image_path = args.input_image
    input_audio_path = args.input_audio
    if "wav" not in input_audio_path:
        logger.warning("MEMO might not generate full-length video for non-wav audio file.")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(
        output_dir,
        f"{os.path.basename(input_image_path).split('.')[0]}_{os.path.basename(input_audio_path).split('.')[0]}.mp4",
    )

    if os.path.exists(output_video_path):
        logger.info(f"Output file {output_video_path} already exists. Skipping inference.")
        return

    generator = torch.manual_seed(args.seed)

    logger.info(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)

    # Determine model paths
    if config.model_name_or_path == "memoavatar/memo":
        logger.info(
            f"The MEMO model will be downloaded from Hugging Face to the default cache directory. The models for face analysis and vocal separation will be downloaded to {config.misc_model_dir}."
        )

        face_analysis = os.path.join(config.misc_model_dir, "misc/face_analysis")
        os.makedirs(face_analysis, exist_ok=True)
        for model in [
            "1k3d68.onnx",
            "2d106det.onnx",
            "face_landmarker_v2_with_blendshapes.task",
            "genderage.onnx",
            "glintr100.onnx",
            "scrfd_10g_bnkps.onnx",
        ]:
            if not os.path.exists(os.path.join(face_analysis, 'models', model)):
                logger.info(f"Downloading {model} to {face_analysis}")
                os.system(
                    f"wget -O {face_analysis}/models/{model} https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/{model}?download=true"
                )
                # Check if the download was successful
                if not os.path.exists(face_analysis):
                    raise RuntimeError(f"Failed to download {model} to {face_analysis}")
                # filesize
                if os.path.getsize(face_analysis) < 1024*1024:
                    raise RuntimeError(f"{face_analysis} file seems incorrect (too small), delete it and retry.")
        logger.info(f"Use face analysis models from {face_analysis}")

        vocal_separator = os.path.join(config.misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
        if os.path.exists(vocal_separator):
            logger.info(f"Vocal separator {vocal_separator} already exists. Skipping download.")
        else:
            logger.info(f"Downloading vocal separator to {vocal_separator}")
            os.makedirs(os.path.dirname(vocal_separator), exist_ok=True)
            os.system(
                f"wget -O {os.path.dirname(vocal_separator)}/Kim_Vocal_2.onnx https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx?download=true"
            )
    else:
        logger.info(f"Loading manually specified model path: {config.model_name_or_path}")
        face_analysis = os.path.join(config.model_name_or_path, "misc/face_analysis")
        vocal_separator = os.path.join(config.model_name_or_path, "misc/vocal_separator/Kim_Vocal_2.onnx")

    # Set up device and weight dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32
    logger.info(f"Inference dtype: {weight_dtype}")

    logger.info(f"Processing image {input_image_path}")
    img_size = (config.resolution, config.resolution)
    pixel_values, face_emb = preprocess_image(
        face_analysis_model=face_analysis,
        image_path=input_image_path,
        image_size=config.resolution,
    )

    logger.info(f"Processing audio {input_audio_path}")
    cache_dir = os.path.join(output_dir, "audio_preprocess")
    os.makedirs(cache_dir, exist_ok=True)
    input_audio_path = resample_audio(
        input_audio_path,
        os.path.join(cache_dir, f"{os.path.basename(input_audio_path).split('.')[0]}-16k.wav"),
    )
    audio_emb, audio_length = preprocess_audio(
        wav_path=input_audio_path,
        num_generated_frames_per_clip=config.num_generated_frames_per_clip,
        fps=config.fps,
        wav2vec_model=config.wav2vec,
        vocal_separator_model=vocal_separator,
        cache_dir=cache_dir,
        device=device,
    )

    logger.info("Processing audio emotion")
    audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
        model=config.model_name_or_path,
        wav_path=input_audio_path,
        emotion2vec_model=config.emotion2vec,
        audio_length=audio_length,
        device=device,
    )

    logger.info("Loading models")
    vae = AutoencoderKL.from_pretrained(config.vae).to(device=device, dtype=weight_dtype)
    reference_net = UNet2DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="reference_net", use_safetensors=True
    )
    diffusion_net = UNet3DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="diffusion_net", use_safetensors=True
    )
    image_proj = ImageProjModel.from_pretrained(
        config.model_name_or_path, subfolder="image_proj", use_safetensors=True
    )
    audio_proj = AudioProjModel.from_pretrained(
        config.model_name_or_path, subfolder="audio_proj", use_safetensors=True
    )

    vae.requires_grad_(False).eval()
    reference_net.requires_grad_(False).eval()
    diffusion_net.requires_grad_(False).eval()
    image_proj.requires_grad_(False).eval()
    audio_proj.requires_grad_(False).eval()

    # Enable memory-efficient attention for xFormers
    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.info(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            reference_net.enable_xformers_memory_efficient_attention()
            diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Create inference pipeline
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline = VideoPipeline(
        vae=vae,
        reference_net=reference_net,
        diffusion_net=diffusion_net,
        scheduler=noise_scheduler,
        image_proj=image_proj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    video_frames = []
    num_clips = audio_emb.shape[0] // config.num_generated_frames_per_clip
    for t in tqdm(range(num_clips), desc="Generating video clips"):
        if len(video_frames) == 0:
            # Initialize the first past frames with reference image
            past_frames = pixel_values.repeat(config.num_init_past_frames, 1, 1, 1)
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
        else:
            past_frames = video_frames[-1][0]
            past_frames = past_frames.permute(1, 0, 2, 3)
            past_frames = past_frames[0 - config.num_past_frames :]
            past_frames = past_frames * 2.0 - 1.0
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = (
            audio_emb[
                t
                * config.num_generated_frames_per_clip : min(
                    (t + 1) * config.num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]
            .unsqueeze(0)
            .to(device=audio_proj.device, dtype=audio_proj.dtype)
        )
        audio_tensor = audio_proj(audio_tensor)

        audio_emotion_tensor = audio_emotion[
            t
            * config.num_generated_frames_per_clip : min(
                (t + 1) * config.num_generated_frames_per_clip, audio_emb.shape[0]
            )
        ]

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            audio_emotion=audio_emotion_tensor,
            emotion_class_num=num_emotion_classes,
            face_emb=face_emb,
            width=img_size[0],
            height=img_size[1],
            video_length=config.num_generated_frames_per_clip,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
        )

        video_frames.append(pipeline_output.videos)

    video_frames = torch.cat(video_frames, dim=2)
    video_frames = video_frames.squeeze(0)
    video_frames = video_frames[:, :audio_length]

    tensor_to_video(video_frames, output_video_path, input_audio_path, fps=config.fps)


if __name__ == "__main__":
    main()
