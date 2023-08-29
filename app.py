from potassium import Potassium, Request, Response
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
import torch
import base64
from io import BytesIO
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():


    base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    #base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    context = {
        "model": base,
        "refiner": refiner,
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    refiner = context.get("refiner")
    prompt = request.json.get("prompt")
   
   


    '''
    n_steps = 40
    high_noise_frac = 0.8
    model.unet = torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)
    
    image = model(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
    image=image,
    ).images[0]
    '''
    
    image = model(prompt=prompt).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format='JPEG',quality=80)
    img_str = base64.b64encode(buffered.getvalue())

    # You could also consider writing this image to S3
    # and returning the S3 URL instead of the image data
    # for a slightly faster response time

    return Response(
        json = {"output": str(img_str, "utf-8")}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
