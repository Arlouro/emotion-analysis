from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
from collections import defaultdict

analyzer = SentimentIntensityAnalyzer()

input_file = "./vader/data/llm_comparative_analysis.csv"
output_file = "./vader/results/llm_comparative_analysis_results.csv"

rows = []
model_sums = defaultdict(float)
model_counts = defaultdict(int)

curr_style = ""
curr_painting = ""
curr_model = ""

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'vader_compound_avg']
    
    for row in reader:
        if row['Style'].strip(): curr_style = row['Style'].strip()
        if row['Painting'].strip(): curr_painting = row['Painting'].strip()
        if row['Model'].strip(): curr_model = row['Model'].strip()
        
        text = row['Explanation']
        sentiment = analyzer.polarity_scores(text)
        
        row['vader_neg'] = sentiment['neg']
        row['vader_neu'] = sentiment['neu']
        row['vader_pos'] = sentiment['pos']
        row['vader_compound'] = sentiment['compound']
        
        group_key = (curr_style, curr_painting, curr_model)
        row['_group_key'] = group_key 
        
        model_sums[group_key] += sentiment['compound']
        model_counts[group_key] += 1
        
        rows.append(row)

with open(output_file, 'w', newline='', encoding='utf-8') as g:
    clean_fieldnames = [f for f in fieldnames if f != '_group_key']
    writer = csv.DictWriter(g, fieldnames=clean_fieldnames)
    writer.writeheader()

    for row in rows:
        group_key = row['_group_key']
        
        if str(row['Iteration']) == '1':
            avg = model_sums[group_key] / model_counts[group_key]
            row['vader_compound_avg'] = round(avg, 4)
        else:
            row['vader_compound_avg'] = "" 
            
        del row['_group_key']
        writer.writerow(row)