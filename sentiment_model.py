# sentiment_model.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Vader
vader_analyzer = SentimentIntensityAnalyzer()

# Hugging Face pipeline
hf_pipeline = pipeline("sentiment-analysis")

def vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def hf_sentiment(text):
    result = hf_pipeline(text)[0]
    return result["label"]
