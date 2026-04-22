import csv
import spacy
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configuration
INPUT_FILE = "./vader/data/llm_comparative_analysis.csv"
OUTPUT_FILE = "./vader/results/llm_comparative_analysis_results.csv"
SPACY_METRICS_FILE = "./vader/results/model_spacy_metrics.csv"
PAINTING_METRICS_FILE = "./vader/results/avg_vader_metrics_per_painting.csv"
MODEL_PAINTING_VADER_METRICS = "./vader/results/model_painting_vader_metrics.csv"


CLEAN_ARTEMIS_FILE = "./vader/results/clean_artemis.csv"
ARTEMIS_DATASET = "./vader/data/artemis_dataset.csv"
ARTEMIS_METRICS_FILE = "./vader/results/artemis_metrics.csv"
ARTEMIS_METRICS_PER_PAINTING_FILE = "./vader/results/artemis_metrics_per_painting.csv"

POS_METRICS = ['Words', 'Nouns', 'Pronouns', 'Adjectives', 'Adpositions', 'Verbs']
ARTEMIS_PAINTING_SELECTION = [
    "felix-vallotton_paul-verlaine-1891",
    "robert-brackman_flowers-for-jennifer",
    "vasily-vereshchagin_retired-butler-1888",
    "henri-fantin-latour_mademoiselle-de-fitz-james-1867", 
    "joseph-wright_vesuvius-from-posillipo",               
    "gustave-courbet_in-the-woods",                        
    "vincent-van-gogh_country-churchyard-and-old-church-tower-1885(1)", 
    "john-constable_dedham-lock-and-mill-1818",            
    "konstantin-somov_lovers",                             
    "david-teniers-the-younger_smokers-in-an-interior",    
    "marc-chagall_in-the-night-1943",                      
    "jamie-wyeth_if-once-you-have-slept-on-an-island-1996", 
    "brice-marden_sea-painting-i-1974",                    
    "frank-bowling_lemongrass-blackpepper-bush",           
    "brice-marden_cyprian-evocation",                      
    "paul-brach_corona-i-1995"
]

def initialize_tools():
    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
    return analyzer, nlp

def get_linguistic_metrics(text, nlp):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    
    return {
        'Words': len(tokens),
        'Nouns': len([t for t in tokens if t.pos_ in ['NOUN', 'PROPN']]),
        'Pronouns': len([t for t in tokens if t.pos_ == 'PRON']),
        'Adjectives': len([t for t in tokens if t.pos_ == 'ADJ']),
        'Adpositions': len([t for t in tokens if t.pos_ == 'ADP']),
        'Verbs': len([t for t in tokens if t.pos_ in ['VERB', 'AUX']])
    }

def process_explanations(input_file, analyzer, nlp):
    rows = []
    group_sums = defaultdict(lambda: defaultdict(float))
    group_counts = defaultdict(int)
    model_overall_sums = defaultdict(lambda: defaultdict(float))
    model_overall_counts = defaultdict(int)
    painting_sums = defaultdict(lambda: defaultdict(float))
    painting_counts = defaultdict(int)

    curr_style, curr_painting, curr_model = "", "", ""

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        new_fields = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'vader_compound_avg']
        for m in POS_METRICS:
            new_fields.extend([m, f'{m}_avg'])
        fieldnames = reader.fieldnames + new_fields

        for row in reader:
            # Handle sparse CSV format
            if row['Style'].strip(): curr_style = row['Style'].strip()
            if row['Painting'].strip(): curr_painting = row['Painting'].strip()
            if row['Model'].strip(): curr_model = row['Model'].strip()
            
            text = row['Explanation']
            group_key = (curr_style, curr_painting, curr_model)
            painting_key = (curr_style, curr_painting)

            # Internal keys for aggregation
            row['_group_key'] = group_key 
            row['_painting_key'] = painting_key
            row['_model_key'] = curr_model 
            
            group_counts[group_key] += 1
            model_overall_counts[curr_model] += 1
            painting_counts[painting_key] += 1       

            # VADER Sentiment Analysis
            sentiment = analyzer.polarity_scores(text)
            row.update({
                'vader_neg': sentiment['neg'],
                'vader_neu': sentiment['neu'],
                'vader_pos': sentiment['pos'],
                'vader_compound': sentiment['compound']
            })
            
            group_sums[group_key]['vader_compound'] += sentiment['compound']
            group_sums[group_key]['vader_neg'] += sentiment['neg']
            group_sums[group_key]['vader_neu'] += sentiment['neu']
            group_sums[group_key]['vader_pos'] += sentiment['pos']
            
            for k in ['neg', 'neu', 'pos', 'compound']:
                painting_sums[painting_key][f'vader_{k}'] += sentiment[k]
            
            # Linguistic Analysis
            counts = get_linguistic_metrics(text, nlp)
            for m in POS_METRICS:
                row[m] = counts[m]
                group_sums[group_key][m] += counts[m]
                model_overall_sums[curr_model][m] += counts[m]            
            
            rows.append(row)

    return rows, group_sums, group_counts, model_overall_sums, model_overall_counts, painting_sums, painting_counts, fieldnames

def save_detailed_results(output_file, rows, group_sums, group_counts, fieldnames):
    with open(output_file, 'w', newline='', encoding='utf-8') as g:
        clean_fieldnames = [f for f in fieldnames if not f.startswith('_')]
        writer = csv.DictWriter(g, fieldnames=clean_fieldnames)
        writer.writeheader()

        for row in rows:
            gk = row['_group_key']
            is_first = str(row['Iteration']) == '1'
            
            row['vader_compound_avg'] = round(group_sums[gk]['vader_compound'] / group_counts[gk], 4) if is_first else ""
            for m in POS_METRICS:
                row[f'{m}_avg'] = round(group_sums[gk][m] / group_counts[gk], 2) if is_first else ""
            
            writer.writerow({k: v for k, v in row.items() if not k.startswith('_')})

def save_model_summary(output_file, model_overall_sums, model_overall_counts):
    with open(output_file, 'w', newline='', encoding='utf-8') as h:
        fields = ['Model'] + [f'Avg_{m}' for m in POS_METRICS]
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()

        for model_name in sorted(model_overall_counts.keys()):
            count = model_overall_counts[model_name]
            metrics_row = {'Model': model_name}
            for m in POS_METRICS:
                metrics_row[f'Avg_{m}'] = round(model_overall_sums[model_name][m] / count, 2)
            writer.writerow(metrics_row)

def save_painting_avg_vader_metrics(output_file, painting_sums, painting_counts):
    with open(output_file, 'w', newline='', encoding='utf-8') as i:
        fields = ['Style', 'Painting', 'Avg_Neg', 'Avg_Neu', 'Avg_Pos', 'Avg_Compound']
        writer = csv.DictWriter(i, fieldnames=fields)
        writer.writeheader()

        for (style, painting) in sorted(painting_counts.keys()):
            count = painting_counts[(style, painting)]
            sums = painting_sums[(style, painting)]
            writer.writerow({
                'Style': style,
                'Painting': painting,
                'Avg_Neg': round(sums['vader_neg'] / count, 4),
                'Avg_Neu': round(sums['vader_neu'] / count, 4),
                'Avg_Pos': round(sums['vader_pos'] / count, 4),
                'Avg_Compound': round(sums['vader_compound'] / count, 4)
            })


def save_model_painting_vader_metrics(output_file, group_sums, group_counts):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fields = ['Style', 'Painting', 'Model', 'Avg_Neg', 'Avg_Neu', 'Avg_Pos', 'Avg_Compound']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for (style, painting, model) in sorted(group_counts.keys()):
            count = group_counts[(style, painting, model)]
            sums = group_sums[(style, painting, model)]
            writer.writerow({
                'Style': style,
                'Painting': painting,
                'Model': model,
                'Avg_Neg': round(sums.get('vader_neg', 0) / count, 4),
                'Avg_Neu': round(sums.get('vader_neu', 0) / count, 4),
                'Avg_Pos': round(sums.get('vader_pos', 0) / count, 4),
                'Avg_Compound': round(sums.get('vader_compound', 0) / count, 4)
            })
    

def filter_artemis(input_file, output_file, selection):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        filtered_rows = []
        for row in reader:
            if row['painting'] in selection:
                filtered_rows.append(row)

        with open(output_file, 'w', newline='', encoding='utf-8') as g:
            writer = csv.DictWriter(g, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)
    
def process_explanations_artemis(input_file, analyzer, nlp):
    rows = []
    painting_sums = defaultdict(lambda: defaultdict(float))
    painting_counts = defaultdict(int)

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        new_fields = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'vader_compound_avg']
        for m in POS_METRICS:
            new_fields.append(m)
        fieldnames = reader.fieldnames + new_fields

        for row in reader:
            text = row['utterance']
            painting_key = (row['art_style'], row['painting'])
            
            painting_counts[painting_key] += 1       

            # VADER Sentiment Analysis
            sentiment = analyzer.polarity_scores(text)
            row.update({
                'vader_neg': sentiment['neg'],
                'vader_neu': sentiment['neu'],
                'vader_pos': sentiment['pos'],
                'vader_compound': sentiment['compound']
            })
            
            for k in ['neg', 'neu', 'pos', 'compound']:
                painting_sums[painting_key][f'vader_{k}'] += sentiment[k]
            
            # Linguistic Analysis
            counts = get_linguistic_metrics(text, nlp)
            for m in POS_METRICS:
                row[m] = counts[m]
            
            rows.append(row)

    for row in rows:
        pk = (row['art_style'], row['painting'])
        count = painting_counts[pk]
        row['vader_compound_avg'] = round(painting_sums[pk]['vader_compound'] / count, 4)

    rows.sort(key=lambda x: (x['painting'], x['repetition']))

    return rows, painting_sums, painting_counts, fieldnames

def save_artemis_results(output_file, rows, fieldnames):
    with open(output_file, 'w', newline='', encoding='utf-8') as g:
        writer = csv.DictWriter(g, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_artemis_paiting_avg_vader_metrics(output_file, painting_sums, painting_counts):
    with open(output_file, 'w', newline='', encoding='utf-8') as i:
        fields = ['Style', 'Painting', 'Avg_Neg', 'Avg_Neu', 'Avg_Pos', 'Avg_Compound']
        writer = csv.DictWriter(i, fieldnames=fields)
        writer.writeheader()

        for (style, painting) in sorted(painting_counts.keys()):
            count = painting_counts[(style, painting)]
            sums = painting_sums[(style, painting)]
            writer.writerow({
                'Style': style,
                'Painting': painting,
                'Avg_Neg': round(sums['vader_neg'] / count, 4),
                'Avg_Neu': round(sums['vader_neu'] / count, 4),
                'Avg_Pos': round(sums['vader_pos'] / count, 4),
                'Avg_Compound': round(sums['vader_compound'] / count, 4)
            })
   

def main():
    analyzer, nlp = initialize_tools()
    
    results = process_explanations(INPUT_FILE, analyzer, nlp)
    rows, g_sums, g_counts, m_sums, m_counts, p_sums, p_counts, fieldnames = results

    #save_detailed_results(OUTPUT_FILE, rows, g_sums, g_counts, fieldnames)

    #save_model_summary(SPACY_METRICS_FILE, m_sums, m_counts)

    #save_painting_avg_vader_metrics(PAINTING_METRICS_FILE, p_sums, p_counts)

    save_model_painting_vader_metrics(MODEL_PAINTING_VADER_METRICS, g_sums, g_counts)

    #filter_artemis(ARTEMIS_DATASET, CLEAN_ARTEMIS_FILE, ARTEMIS_PAINTING_SELECTION)

    results_artemis = process_explanations_artemis(CLEAN_ARTEMIS_FILE, analyzer, nlp)

    rows_artemis, painting_sums_artemis, painting_counts_artemis, fieldnames_artemis = results_artemis

    save_artemis_results(ARTEMIS_METRICS_FILE, rows_artemis, fieldnames_artemis)

    save_artemis_paiting_avg_vader_metrics(ARTEMIS_METRICS_PER_PAINTING_FILE, painting_sums_artemis, painting_counts_artemis)


if __name__ == "__main__":
    main()
