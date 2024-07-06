from transformers import pipeline
classify_sentiment = pipeline("sentiment-analysis")
summarize_text = pipeline("summarization")


def summarize(text):
    print(f"summarization={summarize_text(text)}")


summarize(
    '''
    "I am the Bone of my Sword
    Steel is my Body and Fire is my Blood.
    I have created over a Thousand Blades,
    Unknown to Death,
    Nor known to Life.
    Have withstood Pain to create many Weapons
    Yet those Hands will never hold Anything.
    So, as I Pray--
    Unlimited Blade Works
    ''')
