from __future__ import annotations

import os
# Force spawn method for safe multiprocessing with CUDA
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_USE_V1"] = "0"

from typing import Any
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from utils import extract_out_payload

def load_processor(model_path: str):
    return Qwen3OmniMoeProcessor.from_pretrained(model_path)

def init_llm(
    model: str,
    *,
    dtype: str = "bfloat16",
    kv_cache_dtype: str = "fp8",
    gpu_memory_utilization: float = 0.75,
    tensor_parallel_size: int = 1,
    limit_mm_per_prompt: dict[str, int] | None = None,
    max_num_seqs: int = 4,
    max_model_len: int = 8192,
    seed: int = 1234,
    trust_remote_code: bool = True,
    model_impl: str | None = None,
):
    kwargs = dict(
        model=model,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        # Важно: limit_mm_per_prompt должен позволять multiple audios для few-shot
        limit_mm_per_prompt=limit_mm_per_prompt or {"audio": 10, "image": 0, "video": 0},
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        seed=seed,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    if model_impl is not None:
        kwargs["model_impl"] = model_impl
    return LLM(**kwargs)

def init_sampling(
    *,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 2048,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

def _flatten_and_convert_to_tensors(data) -> list[torch.Tensor]:
    """
    1. Рекурсивно выравнивает вложенные списки (list/tuple).
    2. Конвертирует numpy.ndarray в torch.Tensor.
    Возвращает плоский список тензоров.
    """
    flat = []
    if isinstance(data, (list, tuple)):
        for item in data:
            flat.extend(_flatten_and_convert_to_tensors(item))
    else:
        # Базовый случай: элемент данных
        if data is not None:
            if isinstance(data, np.ndarray):
                # Явная конвертация numpy -> tensor
                flat.append(torch.from_numpy(data))
            elif torch.is_tensor(data):
                flat.append(data)
            else:
                # На случай других типов пробуем обернуть
                try:
                    flat.append(torch.tensor(data))
                except:
                    pass
    return flat

def _process_multi_modal_data(audios, images, videos):
    multi_modal_data = {}
    
    # --- AUDIO PROCESSING ---
    if audios is not None:
        # Получаем плоский список Тензоров
        tensor_list = _flatten_and_convert_to_tensors(audios)
        
        if tensor_list:
            # БЯМС! Исправление: передаем СПИСОК тензоров.
            # Не делаем torch.cat(), чтобы vLLM видел каждое аудио отдельно
            # и мог корректно сопоставить их с токенами <|audio_start|> в промпте.
            multi_modal_data["audio"] = tensor_list
    
    # Обработка изображений (на будущее, если будут)
    if images is not None:
        multi_modal_data["image"] = images

    if videos is not None:
        multi_modal_data["video"] = videos
            
    return multi_modal_data

def build_inputs(
    audio_path: str, 
    prompt_text: str, 
    processor: Qwen3OmniMoeProcessor,
    ) -> dict[str, Any]:
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": prompt_text},
        ],
    }]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    multi_modal_data = _process_multi_modal_data(audios, images, videos)

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }

    return inputs

def build_few_shot_inputs(
    audio_path: str, 
    prompt: str, 
    text: str,
    processor: Qwen3OmniMoeProcessor,
    few_shot_examples: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
    
    messages = []
    user_text_template = "Транскрипт для разметки: \"{example}\"\nВывод:"
    
    prompt_added = False

    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples):
            user_text = user_text_template.format(example=ex["text"])
            
            if i == 0:
                user_text = prompt + "\n" + user_text
                prompt_added = True
                
            user_content = [
                {"type": "audio", "audio": ex["audio"]},
                {"type": "text",  "text": user_text}
            ]
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["response"]})
            
    target_text = user_text_template.format(example=text)

    if not prompt_added:
        target_text = prompt + "\n" + target_text

    messages.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": target_text},
        ],
    })

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    # Здесь теперь вернется список [Tensor, Tensor, Tensor]
    multi_modal_data = _process_multi_modal_data(audios, images, videos)

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }

    return inputs

def generate_texts(llm: LLM, inputs_list: list[dict[str, Any]], sampling: SamplingParams) -> list[str]:
    outs = llm.generate(inputs_list, sampling_params=sampling, use_tqdm=True)
    return [extract_out_payload(o.outputs[0].text) for o in outs]