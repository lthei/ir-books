import re
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)   # needed for wordnet lemma coverage in NLTK >= 3.7

from nltk.corpus import stopwords, wordnet

STOPWORDS = set(stopwords.words("english"))

# maximum synonyms to collect per query token — kept small so the
# query doesn't balloon and start matching unrelated documents
MAX_SYNONYMS_PER_TOKEN = 3


def simple_tokenize(text):
    """Lowercase, strip non-alpha characters, and remove stopwords and short tokens."""
    # adapted from the lab notebook
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


def _get_synonyms(word):
    """Return up to MAX_SYNONYMS_PER_TOKEN WordNet synonyms for a single word."""
    synonyms = []
    seen = set()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            clean = lemma.lower().replace("_", " ")

            # skip multi-word expressions — the tokenizer would split them anyway
            if " " in clean:
                continue

            # skip the original word itself and anything already collected
            if clean == word or clean in seen:
                continue

            # apply the same quality filters as simple_tokenize: no stopwords, no very short tokens
            if clean in STOPWORDS or len(clean) <= 2:
                continue

            seen.add(clean)
            synonyms.append(clean)

            if len(synonyms) >= MAX_SYNONYMS_PER_TOKEN:
                return synonyms

    return synonyms


def expand_query(query):
    """Append WordNet synonyms to a query string and return the expanded version."""
    # tokenise the query the same way simple_tokenize does, so we only
    # expand tokens that would actually reach the index / BM25 / encoder
    text = query.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

    # collect synonyms, making sure we never re-add a word already in the query
    already_present = set(tokens)
    additions = []

    for token in tokens:
        for syn in _get_synonyms(token):
            if syn not in already_present:
                already_present.add(syn)
                additions.append(syn)

    if not additions:
        return query  # WordNet found nothing useful; return the original unchanged

    return query + " " + " ".join(additions)


if __name__ == "__main__":
    sample = "Attention mechanisms have revolutionized natural language processing tasks."
    print(simple_tokenize(sample))

    samples = [
        "mystery detective murder",
        "dystopian society future",
        "romance love forbidden",
    ]
    for q in samples:
        print(f"original : {q}")
        print(f"expanded : {expand_query(q)}")
        print()
