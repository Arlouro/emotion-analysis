from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import os

analyzer = SentimentIntensityAnalyzer()

input = "./vader/data/llm_comparative_analysis.csv"
output = "./vader/results/llm_comparative_analysis_results.csv"

with open(input, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    
    with open(output, 'w', newline='') as g:
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = row['Explanation']
            sentiment = analyzer.polarity_scores(text)
            row['vader_neg'] = sentiment['neg']
            row['vader_neu'] = sentiment['neu']
            row['vader_pos'] = sentiment['pos']
            row['vader_compound'] = sentiment['compound']
            writer.writerow(row)