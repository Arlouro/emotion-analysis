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
text = "I am not very happy, but I am also not especially sad"
doc = nlp(text)

# print polarity of document, scaled to be between -1, and 1
print(doc._.polarity)
# neg=0.0 neu=0.631 pos=0.369 compound=0.7526

# visualize model prediction
html = asent.visualize(doc, style="prediction")
with open("sentiment_engmodel.html", "w", encoding="utf-8") as f:
    f.write(html)

# visualize the analysis performed by the model:
html2 = asent.visualize(doc[:5], style="analysis")
with open("sentiment_analysis_engmodel.html", "w", encoding="utf-8") as f:
    f.write(html2)
