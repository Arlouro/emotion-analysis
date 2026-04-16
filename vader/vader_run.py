from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

analyzer = SentimentIntensityAnalyzer()

texts = []

with open("./vader/llm_comparative_analysis.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        texts.append(row[5])

# CSV header
with open("./vader/results/sentiment_analysis.csv", "w", encoding="utf-8") as f:
    f.write("Text,Negative,Neutral,Positive,Compound")
    f.write("\n")

results = []

for text in texts:
    sentiment = analyzer.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    
    # save to array
    results.append([text, sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']])

    # Save to txt file
    """ with open("./vader/results/sentiment_analysis.txt", "a", encoding="utf-8") as f:
        f.write(f"Text: {text}\n")
        f.write(f"Sentiment: {sentiment}\n\n") """
    
# Save to CSV file
with open("./vader/results/sentiment_analysis.csv", "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(results)