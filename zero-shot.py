from transformers import pipeline

zero_shot_classify = pipeline("zero-shot-classification")
summarize_text = pipeline("summarization")

classification = zero_shot_classify('This is a free course about baking radioactive cookies.',
                                    candidate_labels=['education', 'politics', 'satire'])
print(classification)