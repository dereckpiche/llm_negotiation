from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B"
        )
help(tokenizer)
help(tokenizer.chat_template)