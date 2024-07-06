from transformers import pipeline
import time
import torch
import sys

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

if len(sys.argv) > 1 and sys.argv[1] == 'quantize=true':
    print("Quantizing model...")
    kwargs = {
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    }
    device = None
else:
    print("Using original (non-quantized) model")
    kwargs = {
        "torch_dtype": torch.bfloat16
    }
    device = "cuda"

pipe = pipeline(
    'text-generation',
    model=model_id,
    model_kwargs=kwargs,
    device=device
)

if next(pipe.model.parameters()).is_cuda:
    print("Model loaded by pipeline is running on CUDA")
else:
    print("Model loaded by pipeline is running on CPU")

messages = [
    {"role": "system",
     "content": 'You are an ominous AI assistant known as "PAL."'},
    {"role": "user", "content": "Please explain why the chicken crossed the road."}
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

before_ts_ms = int(time.time() * 1000)
outputs = pipe(messages, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.8, )
after_ts_ms = int(time.time() * 1000)

assistant_response = outputs[0]['generated_text'][-1]['content']
print(assistant_response)
print(f"executionTime={after_ts_ms - before_ts_ms} ms")
