import base64
from io import BytesIO
from diffusion.inference import ModelInference
from PIL import Image

latent_mean = (-0.2000, -0.2609, -1.2588, 0.7237)
latent_std = (4.1562, 1.8995, 3.7727, 2.5186)
noise_scheduler_params = {'rescale_betas_zero_snr': True, 'beta_start': 0.0000085}

sdxl_model = ModelInference(model_name='precomputed_text_latent_diffusion',
                            local_checkpoint_path="/root/v2-finetune/checkpoints/ep9-ba250-rank0.pt",  # PATH TO CHECKPOINT
                            autoencoder_path="/root/v2-finetune/checkpoints/v1-autoencoder.pt",  # PATH TO AUTOENCODER CHECKPOINT,
                            latent_mean=latent_mean,
                            latent_std=latent_std,
                            text_embed_dim=4096,
                            train_noise_scheduler_params=noise_scheduler_params,
                            inference_noise_scheduler_params=noise_scheduler_params,
                            scheduler_shift_resolution=1024,
                            include_text_encoders=True,
                            text_encoder_dtype='bfloat16',
                            use_xformers=False, 
                            cache_dir="/tmp",)

def generate_image(prompt, negative_prompt, guidance_scale, seed):
    generated_images = sdxl_model.predict([
        {
            'input': {
                'prompt': prompt,
                'negative_prompt': negative_prompt
            },
            'parameters': {
                'height': 1024,
                'width': 1024,
                'seed': int(seed),
                'guidance_scale': guidance_scale,
                'num_images_per_prompt': 1,
            }
        }
    ])
    images = [Image.open(BytesIO(base64.b64decode(img))) for img in generated_images]

    return images[0]

img = generate_image("Marsian on the horseback", "", 7.0, 42)
img.save("generated.png")