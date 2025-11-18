from tinygrad.helpers import fetch, getenv
from examples.llama import MODEL_PARAMS, Transformer
from examples.llama3 import Tokenizer
from extra.models.llama import fix_bf16, convert_from_gguf
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, torch_load

MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)

tokenizer_path =       fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")

model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")

tokenizer = Tokenizer(str(tokenizer_path))

model = Transformer(**MODEL_PARAMS["tiny"]["1B"]["args"], linear=nn.Linear, embedding=nn.Embedding, max_context=MAX_CONTEXT, jit=True)

weights = torch_load(str(model_path))

weights = convert_from_gguf(weights, MODEL_PARAMS["tiny"][model_size]["args"]["n_layers"])


if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, params["args"]["n_layers"], params["args"]["n_heads"], params["args"].get("n_kv_heads", params["args"]["n_heads"]))

weights = fix_bf16(weights)
for _,v in weights.items(): v.realize()

load_state_dict(model, weights, strict=False, consume=True)
