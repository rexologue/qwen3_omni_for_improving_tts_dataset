import argparse
from pathlib import Path

from config import SingleInferenceConfig
from mllm_utils import build_few_shot_inputs, build_inputs, generate_texts, init_llm, init_sampling, load_processor
from prompts import get_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обработка одной пары аудио/транскрипт с PROMPT 3 и YAML-конфигом")
    parser.add_argument("--config", type=Path, required=True, help="Путь до YAML конфига")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SingleInferenceConfig.from_yaml(args.config)

    audio_limit = max(1, len(config.examples) + 1) if config.task == "tag" else 1
    llm = init_llm(
        str(config.model),
        dtype=config.dtype,
        kv_cache_dtype=config.kv_cache_dtype,
        gpu_memory_utilization=config.gpu_mem,
        tensor_parallel_size=config.tp,
        limit_mm_per_prompt={"audio": audio_limit, "image": 0, "video": 0},
        max_num_seqs=config.max_seqs,
        max_model_len=config.max_model_len,
        seed=config.seed,
        trust_remote_code=True,
        model_impl="vllm",
    )
    sampling = init_sampling(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
    )
    processor = load_processor(str(config.model))

    if config.task == "tag":
        prompt_text = get_prompt(config.transcript, prompt_type="tags")
        inputs = build_few_shot_inputs(
            str(config.audio),
            prompt_text,
            config.transcript,
            processor,
            config.examples,
        )
    else:
        prompt_text = get_prompt(config.transcript)
        inputs = build_inputs(str(config.audio), prompt_text, processor)

    result = generate_texts(llm, [inputs], sampling)[0]
    print(result)


if __name__ == "__main__":
    main()
