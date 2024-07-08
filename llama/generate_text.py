import transformers

from argparse import ArgumentParser
import torch

parser = ArgumentParser(prog="LlamaTextGenerator",
                        description="Perform text-generation through inference")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",)

model_id = parser.parse_args().model

print(f"model_id={model_id}")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

output = pipeline("Hey how are you doing today?",
                  max_new_tokens=256,
                  eos_token_id=terminators,
                  do_sample=True,
                  temperature=0.6,
                  top_k=10, )
print(output[-1]['generated_text'])
