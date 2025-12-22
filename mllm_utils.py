from __future__ import annotations

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_USE_V1"] = "0"

from typing import Any
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
        limit_mm_per_prompt=limit_mm_per_prompt or {"audio": 5, "image": 0, "video": 0},
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        seed=seed,
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

def build_inputs(
    audio_path: str, 
    prompt_text: str, 
    processor: Qwen3OmniMoeProcessor,
    ) -> dict[str, Any]:
    """
    Универсальный билдер: берёт audio и ГОТОВЫЙ текст подсказки (prompt_text).
    Возвращает dict для llm.generate().
    """
    messages = []
            
    messages.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": prompt_text},
        ],
    })

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)  # type: ignore

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios

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
    
    # Флаг, чтобы понять, добавили мы уже инструкцию или нет
    prompt_added = False

    # 1. Добавляем примеры в историю
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples):
            user_text = user_text_template.format(example=ex["text"])
            
            # Прикрепляем инструкцию к ПЕРВОМУ примеру
            if i == 0:
                user_text = prompt + "\n" + user_text  # Добавил \n для надежности
                prompt_added = True
                
            user_content = [
                {"type": "audio", "audio": ex["audio"]},
                {"type": "text",  "text": user_text}
            ]
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["response"]})
            
    # 2. Формируем целевой запрос
    target_text = user_text_template.format(example=text)

    # Если инструкция еще не была добавлена (т.е. примеров не было), добавляем её сюда
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
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)  # type: ignore

    inputs: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios

    return inputs


def generate_texts(llm: LLM, inputs_list: list[dict[str, Any]], sampling: SamplingParams) -> list[str]:
    outs = llm.generate(inputs_list, sampling_params=sampling, use_tqdm=False)
    return [extract_out_payload(o.outputs[0].text) for o in outs]
