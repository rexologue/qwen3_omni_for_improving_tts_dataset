from __future__ import annotations

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "0")

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
        limit_mm_per_prompt=limit_mm_per_prompt or {"audio": 1, "image": 0, "video": 0},
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

def build_inputs(audio_path: str, prompt_text: str, processor) -> dict[str, Any]:
    """
    Универсальный билдер: берёт audio и ГОТОВЫЙ текст подсказки (prompt_text).
    Возвращает dict для llm.generate().
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": prompt_text},
        ],
    }]

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
