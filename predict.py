import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
nltk.download('punkt')
sns.set(style="whitegrid")

stopwords = set(stopwords.words('english'))
# Detokenizer combines tokenized elements
detokenizer = TreebankWordDetokenizer()



df = pd.read_json (r'data_train.json')

def clean_description(desc):
    desc = word_tokenize(desc.lower())
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return detokenizer.detokenize(desc)

df["cleaned_text"] = df['text'].apply(clean_description)

word_occurrence = df["cleaned_text"].str.split(expand=True).stack().value_counts()
total_words = sum(word_occurrence)

top_words = word_occurrence[:30]/total_words
ax = sns.barplot(x = top_words.values, y = top_words.index)
ax.set_title("% Occurrence of Most Frequent Words")

plt.show()

