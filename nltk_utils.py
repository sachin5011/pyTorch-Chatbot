import nltk 
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

# nltk.download('punkt')

def tokenize(sentance):
    return word_tokenize(sentance)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(toknized_sentance, all_words):
    sentence_words = [stem(word) for word in toknized_sentance]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in sentence_words:
            bag[idx] = 1
    
    return bag

