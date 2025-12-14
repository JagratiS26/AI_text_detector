import numpy as np
import nltk
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class FeatureEngine:
    def __init__(self, text):
        self.text = str(text) if text else ""
        self.sentences = sent_tokenize(self.text)
        self.words = word_tokenize(self.text)
        self.words_only = [w.lower() for w in self.words if w.isalpha()]
        self.stopwords = set(stopwords.words('english'))

    def get_avg_sentence_length(self):
        if not self.sentences: return 0
        return len(self.words) / len(self.sentences)

    def get_vocab_richness(self):
        if not self.words_only: return 0
        return len(set(self.words_only)) / len(self.words_only)

    def get_burstiness(self):
        lengths = [len(word_tokenize(s)) for s in self.sentences]
        if len(lengths) < 2: return 0
        return np.std(lengths) / (np.mean(lengths) + 1e-6)

    def get_readability(self):
        return textstat.flesch_reading_ease(self.text)

    def get_avg_word_length(self):
        if not self.words_only: return 0
        return np.mean([len(w) for w in self.words_only])

    def get_punctuation_ratio(self):
        punct = sum(1 for c in self.text if c in ".,;:!?")
        return punct / (len(self.text) + 1e-6)

    def get_stopword_ratio(self):
        if not self.words_only: return 0
        return sum(1 for w in self.words_only if w in self.stopwords) / len(self.words_only)

    def extract_all(self):
        return {
            "avg_sent_len": self.get_avg_sentence_length(),
            "vocab_richness": self.get_vocab_richness(),
            "burstiness": self.get_burstiness(),
            "readability": self.get_readability(),
            "avg_word_len": self.get_avg_word_length(),
            "punct_ratio": self.get_punctuation_ratio(),
            "stopword_ratio": self.get_stopword_ratio()
        }
