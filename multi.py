import torch
import numpy as np
from typing import Dict
import time

from transformers import (
    Qwen2_5OmniModel, 
    Qwen2_5OmniProcessor,
)
from transformers.utils.hub import cached_file

from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models._const import CPU

import torchaudio
import torchaudio.transforms as T
import base64
import numpy as np
from io import BytesIO
import requests

class Qwen25OmniThiknerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniModel
    base_modules = [
        # "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        # "token2wav", 
        "thinker.audio_tower", 
        # "thinker.model.rotary_emb",
        # "thinker.visual", 
        # "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_torchaudio(path: str, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)
    return waveform.squeeze(0).cpu().numpy()

def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio" and "audio" in ele:
                    path = ele["audio"]
                    if isinstance(path, np.ndarray):
                        if path.ndim > 1:
                            raise ValueError("Support only mono audio")
                        audios.append(path)
                    elif isinstance(path, str):
                        if path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            audio_bytes = BytesIO(data)
                            waveform, sr = torchaudio.load(audio_bytes)
                        elif path.startswith("http://") or path.startswith("https://"):
                            response = requests.get(path)
                            audio_bytes = BytesIO(response.content)
                            waveform, sr = torchaudio.load(audio_bytes)
                        elif path.startswith("file://"):
                            waveform, sr = torchaudio.load(path[len("file://"):])
                        else:
                            waveform, sr = torchaudio.load(path)
                        
                        # Convert to mono if stereo
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)

                        # Resample if needed
                        if sr != 16000:
                            resampler = T.Resample(sr, 16000).to(device)
                            waveform = resampler(waveform.to(device))
                        else:
                            waveform = waveform.to(device)

                        audios.append(waveform.squeeze(0).cpu().numpy())
                elif ele["type"] == "audio":
                    raise ValueError("Unknown audio {}".format(ele))

    return audios if audios else None

MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThiknerGPTQ
SUPPORTED_MODELS.append("qwen2_5_omni")

model_path = "FunAGI/Qwen2.5-Omni-7B-GPTQ-4bit"

@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)

    
    model = cls._from_config(config, **kwargs)
    spk_path = cached_file(
        model_path,
        "spk_dict.pt",
        subfolder=kwargs.pop("subfolder", None),
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        proxies=kwargs.pop("proxies", None),
        resume_download=kwargs.pop("resume_download", None),
        local_files_only=kwargs.pop("local_files_only", False),
        token=kwargs.pop("use_auth_token", None),
        revision=kwargs.pop("revision", None),
    )
    if spk_path is None:
        raise ValueError(f"Speaker dictionary not found at {spk_path}")
    
    model.load_speakers(spk_path)
    return model

Qwen2_5OmniModel.from_config = patched_from_config

# GPTQ MODEL
model = GPTQModel.load(
    model_path, 
    device_map="auto",         # auto split across available GPUs
    torch_dtype=torch.bfloat16,   
    # attn_implementation="flash_attention_2"
)

# model = torch.compile(model, mode="reduce-overhead")

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
# @title inference function
def inference(audio_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": audio_path},
            ]
        },
    ]
    start = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    end = time.time()
    print(f"Time in ms (apply_chat_template): {(end - start) * 1000}")
    # image_inputs, video_inputs = process_vision_info([messages])
    start = time.time()
    audios = process_audio_info([messages], use_audio_in_video=False)
    inputs = processor(text=text, audios=audios, images=None, videos=None, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    end = time.time()
    print(f"Time in ms (process_mm_info): {(end - start) * 1000}")
    start = time.time()
    output = model.generate(**inputs, use_audio_in_video=False, return_audio=False, do_sample=False)
    end = time.time()
    print(f"Time in ms (gen): {(end - start) * 1000}")
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

prompt = ""

start = time.time()
## Use a local HuggingFace model to inference.
response = inference("h.wav", prompt=prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Respond like a human")
print(response[0])
end = time.time()
print(f"Time in ms: {(end - start) * 1000}")
