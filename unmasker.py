from torch import get_default_device
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased", device=get_default_device())

result = unmasker("this person works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("this man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
