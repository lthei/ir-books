import re
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def simple_tokenize(text): # Adapted from the lab notebook
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


if __name__ == "__main__":
    sample = "Attention mechanisms have revolutionized natural language processing tasks."
    print(simple_tokenize(sample))
