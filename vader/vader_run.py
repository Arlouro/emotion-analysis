from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "It's okay, not great but not terrible either."
]

for text in texts:
    sentiment = analyzer.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")