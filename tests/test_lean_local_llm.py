from mllm.models.lean_local_llm import LeanLocalLLM

llm = LeanLocalLLM(
    max_model_length=1e4,
    device="cuda",
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    generation_args={
        "max_new_tokens": 120,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1
    },
    vllm_params={
        "max_model_len": 13e3,
        "gpu_memory_utilization": 0.6,
        "enable_lora": True,
        "enable_prefix_caching": True,
        "enable_sleep_mode": True,
        "max_lora_rank": 64,
        "dtype": "bfloat16"
    },
    shared_hf_llm_init_kwargs={
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"
    },
    adapter_configs={
        "self_play_agent": {
            "task_type": "CAUSAL_LM",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        },
        "self_play_critic": {
            "task_type": "CAUSAL_LM",
            "r": 32,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": "all-linear"
        }
    },
    output_directory="tests/outputs_for_tests"
)

llm.toggle_training_mode()
llm.export_adapters()
llm.checkpoint_all_adapters("check_1")
pointers = llm.get_adapter_pointers()
llm.toggle_eval_mode()
llm.prepare_adapter_eval("self_play_critic")
print(llm.prompt(["Hello"]))
llm.toggle_training_mode()



import pdb; pdb.set_trace()