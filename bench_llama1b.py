from examples.llama import Transformer, MODEL_PARAMS
from tinygrad.nn.state import get_state_dict
from tinygrad import Tensor
import time

model = Transformer(**MODEL_PARAMS["tiny"]["1B"]["args"])

print("loading weights")
for v in get_state_dict(model).values():
    v.assign(Tensor.empty(*v.shape, dtype=v.dtype))

print("testing...")
tms = [time.perf_counter()]
for i in range(5):
    model(Tensor([[1,2,3,4]]), i).realize()
    tms.append(time.perf_counter())

st = "codegen(0)"
timings = [(tms[i+1]-tms[i])*1000 for i in range(len(tms)-1)]
print(f"{st:15s} mean runtime: {sum(timings)/len(timings):7.2f}ms, runs: ", ", ".join(f'{x:7.2f}' for x in timings))
