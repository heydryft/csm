import asyncio
import uuid
import torch
import os
import threading
import queue
from vllm import AsyncEngineArgs, SamplingParams, AsyncLLMEngine
from transformers import AutoTokenizer
from .decoder import tokens_decoder_sync

class OrpheusModel:
    def __init__(self, model_name='canopylabs/orpheus-tts-0.1-finetune-prod', dtype=torch.bfloat16, **engine_kwargs):
        self.model_name = model_name
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

        # Background event loop
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

        # Initialize tokenizer using the background loop
        future = asyncio.run_coroutine_threadsafe(self.engine.get_tokenizer(), self.loop)
        self.tokenizer = future.result()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _load_tokenizer(self, tokenizer_path):
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")

    def _map_model_params(self, model_name):
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported.")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name

    def _setup_engine(self):
        num_gpus = torch.cuda.device_count()

        def round_to_nearest_even(n):
            floor_even = int(n // 2) * 2
            ceil_even = floor_even + 2
            return floor_even if abs(n - floor_even) < abs(n - ceil_even) else ceil_even

        parallel_tensors = round_to_nearest_even(num_gpus)

        if os.environ.get("CUDA_VISIBLE_DEVICES") or num_gpus == 1:
            parallel_tensors = 1

        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            tensor_parallel_size=parallel_tensors,  # Round down to nearest even number
            num_scheduler_steps=16,
            **self.engine_kwargs
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    def validate_voice(self, voice):
        if voice and voice not in self.engine.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                return self.tokenizer.decode(all_input_ids[0])
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                return self.tokenizer.decode(all_input_ids[0])

    def generate_tokens_sync(self, prompt, voice=None, request_id=None, temperature=0.6, top_p=0.8,
                             max_tokens=8192, stop_token_ids=[49158], repetition_penalty=1.3):
        request_id = request_id or str(uuid.uuid4())
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(
                prompt=prompt_string,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)

        asyncio.run_coroutine_threadsafe(async_producer(), self.loop)

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

    async def generate_tokens_async(self, prompt, voice=None, request_id=None, temperature=0.6, top_p=0.8,
                                    max_tokens=8192, stop_token_ids=[49158], repetition_penalty=1.3):
        request_id = request_id or str(uuid.uuid4())
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(
                prompt=prompt_string,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)

        producer_task = asyncio.create_task(async_producer())

        while True:
            try:
                token = token_queue.get_nowait()
                if token is None:
                    break
                yield token
            except queue.Empty:
                await asyncio.sleep(0)

        await producer_task

    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
