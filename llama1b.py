from tinygrad.helpers import fetch, tqdm, JIT, getenv
from examples.llama import Transformer, MODEL_PARAMS, load
from tinygrad.nn.state import safe_load, load_state_dict
from extra.models.llama import fix_bf16, convert_from_huggingface
from tinygrad import Tensor, nn, dtypes

MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)

tokenizer_path = fetch("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model", "tokenizer.model", subdir="TinyLlama-1.1B-Chat-v1.0")

model_path = fetch("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors", "model.safetensors", subdir="TinyLlama-1.1B-Chat-v1.0")

print(f"Tokenizer Path:\n{tokenizer_path}\n")
print(f"Model path:\n{model_path}\n")

params = MODEL_PARAMS["tiny"]["1B"]

tokenizer = MODEL_PARAMS["tiny"]["tokenizer"](model_file=str(tokenizer_path))

model = Transformer(**params["args"], linear=nn.Linear, max_context=4096, jit=bool(JIT))

weights = safe_load(str(model_path))

if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, params["args"]["n_layers"], params["args"]["n_heads"], params["args"].get("n_kv_heads", params["args"]["n_heads"]))

weights = fix_bf16(weights)

# replace weights in model
load_state_dict(model, weights, strict=False, consume=True)

prompt = "Hello, world!"
start_pos, toks = 0, [tokenizer.bos_id()] + tokenizer.encode(prompt)

while True:
    res = model(Tensor([toks]), 0, 1.9).realize().item()
    toks.append(res)
    print(str(tokenizer.decode([res])), flush=True)
