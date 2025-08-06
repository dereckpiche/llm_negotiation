# engine = VLLMAsyncBackend(
#     tokenizer=None,
#     adapter_paths={},
#     model_name="Qwen/Qwen3-0.6B",
#     max_model_len=100,
#     max_num_seqs=1,
#     dtype="bfloat16",
#     trust_remote_code=True,
# )
# print("DOOOOONE")
# a = 1 / 0
import asyncio

import torch


async def main():
    from mllm.models.inference_backend_vllm import VLLMAsyncBackend
    from mllm.models.lean_local_llm import LeanLocalLLM

    llm = LeanLocalLLM(
        model_name="Qwen/Qwen3-0.6B",
        inference_backend="vllm",
        hf_kwargs={
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "max_memory": {0: "12GiB"},
        },
        inference_backend_init_kwargs={
            "max_model_len": 100,
            "gpu_memory_utilization": 0.4,
            "max_num_batched_tokens": 1000,
            "max_num_seqs": 1,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "enforce_eager": True,
            "max_lora_rank": 128,
        },
        inference_backend_sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 128,
            "top_k": 0,
        },
        adapter_configs={
            "Alice": {
                "task_type": "CAUSAL_LM",
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
            },
            "Bob": {
                "task_type": "CAUSAL_LM",
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
            },
            "Carl": {
                "task_type": "CAUSAL_LM",
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
            },
        },
        output_directory="tests/outputs_for_tests",
    )

    inference_policies = llm.get_inference_policies()
    adapter_modules = llm.get_adapter_modules()
    carl = adapter_modules["Carl"]
    carl_params = carl.parameters()
    carl_inference = inference_policies["base_llm/Carl"]
    alice_inference = inference_policies["base_llm/Alice"]
    bob_inference = inference_policies["base_llm/Bob"]
    tasks = [
        alice_inference([{"role": "user", "content": "Hello, give me the alphabet."}]),
        carl_inference([{"role": "user", "content": "Hello, give me the alphabet."}]),
        bob_inference([{"role": "user", "content": "Hello, give me the alphabet."}]),
    ]
    alice_res, carl_res, bob_res = await asyncio.gather(*tasks)
    print(f"Alice (concurrent): {alice_res}")
    print(f"Carl  (concurrent): {carl_res}")
    print(f"Bob   (concurrent): {bob_res}")

    llm.toggle_training_mode()
    # Corrupt Carl
    for p in carl_params:
        p.requires_grad = False
        p[:] = torch.randn(p.shape).to(p.device)
    llm.export_adapters()
    llm.toggle_eval_mode()
    res = await carl_inference(
        [{"role": "user", "content": "Hello, give me the alphabet."}]
    )
    print(f"Reponse from corrupted Carl: {res}")

    # Normal Bob
    bob_inference = inference_policies["base_llm/Bob"]
    res = await bob_inference(
        [{"role": "user", "content": "Hello, give me the alphabet."}]
    )
    print(f"Reponse from correct Bob: {res}")

    # Constrained Alice
    alice_inference = inference_policies["base_llm/Alice"]
    res = await alice_inference(
        [{"role": "user", "content": "Hello, give me the alphabet."}], regex="(<A>|<B>)"
    )
    print(f"Reponse from constrained Alice: {res}")


if __name__ == "__main__":
    import multiprocessing as mp

    # set BEFORE any CUDA init; vLLM will also try to enforce spawn
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
