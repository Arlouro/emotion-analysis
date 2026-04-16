from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "It's okay, not great but not terrible either.",
    "I am not very happy, but I am also not especially sad"
]

# CSV header
with open("./vader/results/sentiment_analysis.csv", "w", encoding="utf-8") as f:
    f.write("Text,Negative,Neutral,Positive,Compound")

for text in texts:
    sentiment = analyzer.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")

    # Save to txt file
    """ with open("./vader/results/sentiment_analysis.txt", "a", encoding="utf-8") as f:
        f.write(f"Text: {text}\n")
        f.write(f"Sentiment: {sentiment}\n\n") """
    
    # Save to CSV file
    with open("./vader/results/sentiment_analysis.csv", "a", encoding="utf-8") as f:
        f.write(f"\n{text},{sentiment['neg']},{sentiment['neu']},{sentiment['pos']},{sentiment['compound']}")