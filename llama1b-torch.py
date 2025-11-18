from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging
from tinygrad.helpers import Timing, Context, fetch, CPU_COUNT
from tinygrad.device import Device
from tinygrad import Tensor, dtypes
from examples.llama3 import build_transformer
import torch, os

def bench_torch(sample: int):
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_config(config).to("cpu")
    test_inputs = torch.tensor([[1,2,3,4]], device=torch.device("cpu"))
    with Timing("Torch: total ", on_exit=lambda x: f", {sample/(x*1e-9):.3f} tok/s"):
        with torch.no_grad(): model.generate(test_inputs, min_new_tokens=sample, max_new_tokens=sample, attention_mask=torch.ones_like(test_inputs))

def bench_tiny(sample: int):
    model_path = fetch("https://huggingface.co/unsloth/Llama-3.2-1B/resolve/main/model.safetensors", "model.safetensors", "Llama-3.2-1B")
    with Context(CPU_COUNT=8, JIT=1, CPU_LLVM=1):
        model = build_transformer(model_path, model_size="1B", scale_dtype=dtypes.float16, max_context=4)
        input_tokens = Tensor([[1,2,3,4]])
        for i in range(2): model(input_tokens, 0, 1.9).realize()
        with Timing("Tiny: total ", on_exit=lambda x: f", {sample/(x*1e-9):.3f} tok/s"):
            for i in range(sample): model(input_tokens, 0, 1.9).realize()

def bench_tiny_minimal(sample: int):
    return NotImplemented

# Why is Tiny using metal by default???
if __name__ == "__main__":
    #bench_torch(5)
    bench_tiny(5)
