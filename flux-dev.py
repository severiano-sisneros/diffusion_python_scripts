import torch
from diffusers import FluxPipeline 
import gradio as gr

# Load FLUX.1-dev pipeline and enable CUDA
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

# Enable offloading CPU to help with GPU memory management
pipe.enable_model_cpu_offload()

# Load LoRAs
pipe.load_lora_weights('linoyts/yarn_art_flux_1_700_custom', weight_name='pytorch_lora_weights.safetensors', adapter_name="yarn_art")
pipe.load_lora_weights('alvdansen/flux_film_foto', weight_name='araminta_k_flux_film_foto.safetensors', adapter_name="film_foto")
pipe.load_lora_weights('alvdansen/softserve_anime', weight_name='flux_dev_softstyle_araminta_k.safetensors', adapter_name="softserve_anime")

# Function called whenever the "generate" button is clicked 
def generate(positive_prompt, height, width, guidance, steps, seed, lora_select=None, lora_weight=0.0):

  # If seed is set to 0, generate a random seed using the CPU
  if seed == 0:
    seed = torch.Generator("cpu").seed()

  # Print the seed for debug
  print(seed)

  # Select the appropriate lora, if one has been selected
  if lora_select is not None:
    pipe.set_adapters(lora_select)

  # Generate the image
  image = pipe(
      positive_prompt,
      height=height,
      width=width,
      guidance_scale=guidance,
      num_inference_steps=steps,
      generator=torch.Generator("cpu").manual_seed(seed),
      joint_attention_kwargs={"scale": lora_weight},
  ).images[0]

  # Save locally and return the image
  image.save("./flux-schnell.png")
  return "./flux-schnell.png"

# Configure Gradio GUI
with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            positive_prompt = gr.Textbox(lines=3, interactive=True, value="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black dress with a gold leaf pattern and a white apron eating a slice of an apple pie in the kitchen of an old dark victorian mansion with a bright window and very expensive stuff everywhere", label="Prompt")
            width = gr.Slider(minimum=256, maximum=2048, value=1024, step=16, label="width")
            height = gr.Slider(minimum=256, maximum=2048, value=1024, step=16, label="height")
            seed = gr.Slider(minimum=0, maximum=18446744073709551615, value=0, step=1, label="seed (0=random)")
            guidance = gr.Slider(minimum=0, maximum=20, value=3.5, step=0.5, label="guidance")
            steps = gr.Slider(minimum=4, maximum=50, value=4, step=1, label="steps")
            lora_select = gr.Dropdown(["film_foto", "yarn_art", "softserve_anime"])
            lora_strength_model = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.1, label="lora_strength_model")
            generate_button = gr.Button("Generate")
        with gr.Column():
            output_image = gr.Image(label="Generated image", interactive=False)

    generate_button.click(fn=generate, inputs=[positive_prompt, height, width, guidance, steps, seed, lora_select, lora_strength_model], outputs=output_image)

# Launch app
demo.queue().launch(inline=False, share=True, debug=True)
