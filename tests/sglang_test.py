"""
This example demonstrates how to launch the offline engine.
"""

import sglang as sgl

def main():
    import torch

    device0 = torch.device("cuda:0")
    #device1 = torch.device("cuda:1")
    engine0 = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device0)
    #engine1 = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device1)

    
    engine0.generate("What is the capital of France?")
    #engine1.generate("What is the capital of France?")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()