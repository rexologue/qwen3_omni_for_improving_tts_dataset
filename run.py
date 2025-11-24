import os
from mllm_utils import init_llm, init_sampling, load_processor, build_inputs, generate_texts

MODEL_PATH = "/home/user5/.models/qwen3-omni-awq"
AUDIO = "/home/user5/audio_ref/ref.wav"

def main():
    llm = init_llm(
        MODEL_PATH,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.75,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
        max_num_seqs=4,
        max_model_len=8192,
        seed=1234,
        trust_remote_code=True,
        model_impl="vllm",
    )
    sampling = init_sampling(temperature=0.2, top_p=0.9, top_k=40, max_tokens=2048)
    processor = load_processor(MODEL_PATH)

    prompt_text = "Произведи транскрибацию текста из аудио. Помести ответ в тэги <out> </out>."
    inputs = build_inputs(AUDIO, prompt_text, processor)

    res = generate_texts(llm, [inputs], sampling)[0]
    print(res)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
