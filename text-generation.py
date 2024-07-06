from transformers import pipeline

generate = pipeline('text-generation', model='distilgpt2')
generated = generate(
  'In this course, we will teach you how to',
  max_length=70,
  num_return_sequences=1
)

print(generated)
