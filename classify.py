from transformers import pipeline
classify_sentiment = pipeline("sentiment-analysis")
summarize_text = pipeline("summarization")


def classify(text):
    print(f"text=${text}, classification={classify_sentiment(text)}")


classify("I've been waiting for a HuggingFace course my whole dang life!!!")
classify("I've been waiting for a HuggingFace course my whole life.")
classify("Nuclear Apocalypse")
