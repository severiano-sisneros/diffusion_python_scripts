# Diffusion Python Scripts
These are a collection of scripts I've been using to play with different diffusion models.

## flux-dev.py

Using `diffusers` to experiment with inference using the `flux-dev` model. Using `gradio` for GUI.

To use install the following:

```
sudo apt install python3-pip
pip install torch
pip install diffusers
pip install gradio
pip install transformers
pip install accelerate
pip install sentencepiece
pip install protobuff
pip install peft
```
To run you need a HuggingFace authentication token. 

`HF_TOKEN=<YOUR_TOKEN> python3 flux-dev.py`

This starts a Gradio server you can use to play with the model and various LoRAs.
