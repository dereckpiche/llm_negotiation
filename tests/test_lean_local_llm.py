from mllm.models.lean_local_llm import LeanLocalLLM
import asyncio
import torch

llm = LeanLocalLLM(
    max_model_length=int(1e4),
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_configs={
        "Alice": {
            "task_type": "CAUSAL_LM",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        },
        "Bob": {
            "task_type": "CAUSAL_LM",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        },
        "Carl": {
            "task_type": "CAUSAL_LM",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        },
        # "critic": {
        #     "task_type": "CAUSAL_LM",
        #     "r": 32,
        #     "lora_alpha": 128,
        #     "lora_dropout": 0.0,
        #     "target_modules": "all-linear"
        # }
    },
    output_directory="tests/outputs_for_tests"
)

async def main():
    llm.toggle_training_mode()
    # llm.export_adapters()
    llm.checkpoint_all_adapters("check_1")
    inference_policies = llm.get_inference_policies()
    # training_policies = llm.get_training_policies()
    # carl = training_policies["Carl"]
    # carl_params = carl.parameters()
    carl_inference = inference_policies["llama/Carl"]
    res = await carl_inference([{"role":"user", "content": "Hello, give me the alphabet."}])
    # print(res)
    # for p in carl_params:
    #     p.requires_grad = False
    #     p += 1e5 * torch.randn(p.shape).to(p.device)
    llm.export_adapters()
    # res = await carl_inference([{"role":"user", "content": "Hello, give me the alphabet."}])
    print(res)
    import ipdb; ipdb.set_trace()
asyncio.run(main())
