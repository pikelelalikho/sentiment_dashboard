import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load test data
df = pd.read_csv("test_data.csv")

# Initialize Vader
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize Hugging Face pipeline
hf_pipeline = pipeline("sentiment-analysis")

def vader_predict(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def hf_predict(text):
    result = hf_pipeline(text)[0]["label"]
    if result == "POSITIVE":
        return "Positive"
    elif result == "NEGATIVE":
        return "Negative"
    else:
        return "Neutral"

# Apply predictions
df["vader_pred"] = df["text"].apply(vader_predict)
df["hf_pred"] = df["text"].apply(hf_predict)

# Calculate accuracy
vader_acc = (df["vader_pred"] == df["label"]).mean()
hf_acc = (df["hf_pred"] == df["label"]).mean()

print(f"✅ Vader Accuracy: {vader_acc * 100:.2f}%")
print(f"✅ Hugging Face Accuracy: {hf_acc * 100:.2f}%")

# Save results for your report
df.to_csv("evaluation_results.csv", index=False)
print("✅ Evaluation results saved to evaluation_results.csv")
