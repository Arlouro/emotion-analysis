import csv
import spacy
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

input_file = "./vader/data/llm_comparative_analysis.csv"
output_file = "./vader/results/llm_comparative_analysis_results.csv"
spacy_metrics_file = "./vader/results/model_spacy_metrics.csv"

pos_metrics = ['Words', 'Nouns', 'Pronouns', 'Adjectives', 'Adpositions', 'Verbs']

rows = []
group_sums = defaultdict(lambda: defaultdict(float))
group_counts = defaultdict(int)

model_overall_sums = defaultdict(lambda: defaultdict(float))
model_overall_counts = defaultdict(int)

curr_style = ""
curr_painting = ""
curr_model = ""

# Info Extraction and Grouping
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    new_fields = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'vader_compound_avg']
    for m in pos_metrics:
        new_fields.extend([m, f'{m}_avg'])
    
    fieldnames = reader.fieldnames + new_fields
    
    for row in reader:
        if row['Style'].strip(): curr_style = row['Style'].strip()
        if row['Painting'].strip(): curr_painting = row['Painting'].strip()
        if row['Model'].strip(): curr_model = row['Model'].strip()
        
        text = row['Explanation']
        group_key = (curr_style, curr_painting, curr_model)
        painting_key = (curr_style, curr_painting)

        row['_group_key'] = group_key 
        row['_model_key'] = curr_model 
        row['_painting_key'] = curr_painting
        
        group_counts[group_key] += 1
        model_overall_counts[curr_model] += 1        

        sentiment = analyzer.polarity_scores(text)
        row['vader_neg'] = sentiment['neg']
        row['vader_neu'] = sentiment['neu']
        row['vader_pos'] = sentiment['pos']
        row['vader_compound'] = sentiment['compound']
        
        group_sums[group_key]['vader_compound'] += sentiment['compound']
        model_overall_sums[curr_model]['vader_compound'] += sentiment['compound']
        
        doc = nlp(text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        
        counts = {
            'Words': len(tokens),
            'Nouns': len([t for t in tokens if t.pos_ in ['NOUN', 'PROPN']]),
            'Pronouns': len([t for t in tokens if t.pos_ == 'PRON']),
            'Adjectives': len([t for t in tokens if t.pos_ == 'ADJ']),
            'Adpositions': len([t for t in tokens if t.pos_ == 'ADP']),
            'Verbs': len([t for t in tokens if t.pos_ in ['VERB', 'AUX']])
        }
        
        for m in pos_metrics:
            row[m] = counts[m]
            group_sums[group_key][m] += counts[m]
            model_overall_sums[curr_model][m] += counts[m]            
        rows.append(row)

# AVG Vader metrics per model
with open(output_file, 'w', newline='', encoding='utf-8') as g:
    clean_fieldnames = [f for f in fieldnames if f not in ['_group_key', '_model_key']]
    writer = csv.DictWriter(g, fieldnames=clean_fieldnames)
    writer.writeheader()

    for row in rows:
        gk = row['_group_key']
        is_first = str(row['Iteration']) == '1'
        
        row['vader_compound_avg'] = round(group_sums[gk]['vader_compound'] / group_counts[gk], 4) if is_first else ""
            
        for m in pos_metrics:
            row[f'{m}_avg'] = round(group_sums[gk][m] / group_counts[gk], 2) if is_first else ""
        
        del row['_group_key']
        del row['_model_key']
        writer.writerow(row)

# AVG Spacy metrics per model
with open(spacy_metrics_file, 'w', newline='', encoding='utf-8') as h:
    spacy_metrics_fields = ['Model'] + [f'Avg_{m}' for m in pos_metrics]
    writer = csv.DictWriter(h, fieldnames=spacy_metrics_fields)
    writer.writeheader()

    for model_name in sorted(model_overall_counts.keys()):
        count = model_overall_counts[model_name]
        
        spacy_metrics_row = {'Model': model_name}
        
        for m in pos_metrics:
            m_avg = model_overall_sums[model_name][m] / count
            spacy_metrics_row[f'Avg_{m}'] = round(m_avg, 2)
            
        writer.writerow(spacy_metrics_row)

print(f"Detailed results saved to: {output_file}")
print(f"Model summary saved to: {spacy_metrics_file}")