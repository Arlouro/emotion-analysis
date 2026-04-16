import spacy
from spacy import displacy
import asent

# create spacy pipeline
""" nlp = spacy.blank('en')
nlp.add_pipe('sentencizer') """

# Load the large English model, which includes word vectors
nlp = spacy.load("en_core_web_lg")


# add the rule-based sentiment model
nlp.add_pipe("asent_en_v1")

# try an example
texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "It's okay, not great but not terrible either.",
    "I am not very happy, but I am also not especially sad"
]

for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    print(f"Polarity: {doc._.polarity}")

# visualize model prediction
html = asent.visualize(doc, style="prediction")
with open("./asent/sentiment_engmodel.html", "w", encoding="utf-8") as f:
    f.write(html) 

# visualize the analysis performed by the model:
html2 = asent.visualize(doc, style="analysis")
with open("./asent/sentiment_analysis_engmodel.html", "w", encoding="utf-8") as f:
    f.write(html2)
