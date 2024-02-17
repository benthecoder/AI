import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option("display.max_rows", None)


def tokenize(text: str) -> List[str]:
    """Tokenize the input text by removing punctuation and splitting into words."""
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    tokens = cleaned_text.lower().split()
    return tokens


def calculate_word_frequencies(document: List[str]) -> Dict[str, int]:
    """Calculate the frequency of each word in a document."""
    frequencies = {}
    for word in document:
        frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def calculate_tf(word_counts: Dict[str, int], document_length: int) -> Dict[str, float]:
    """Calculate term frequency for each word in a document."""

    tf_dict = {
        word: count / float(document_length) for word, count in word_counts.items()
    }

    return tf_dict


def calculate_idf(documents_word_counts: List[Dict[str, int]]) -> Dict[str, float]:
    """Calculate inverse document frequency for each word across all documents."""
    N = len(documents_word_counts)
    idf_dict = {}
    unique_words = set(word for doc in documents_word_counts for word in doc)

    for word in unique_words:
        # count number of docs containing the word
        doc_containing_word = sum(
            word in document for document in documents_word_counts
        )

        idf_dict[word] = np.log10((N + 1) / (doc_containing_word + 1))

    return idf_dict


def calculate_tfidf(
    tf_dict: Dict[str, float], idf_dict: Dict[str, float]
) -> Dict[str, float]:
    """Calculate TF-IDF for each word in a document."""

    tfidf_dict = {word: tf_val * idf_dict[word] for word, tf_val in tf_dict.items()}

    return tfidf_dict


def visualize_tfidf(tfidf_matrix: pd.DataFrame):
    """Visualize the TF-IDF matrix using a heatmap."""
    plt.figure(figsize=(10, 10))
    sns.heatmap(tfidf_matrix, annot=True, cmap="YlGnBu")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    # seneca
    sentences = [
        "Life, if well lived, is long enough.",
        "Your time is limited, so don't waste it living someone else's life.",
    ]

    documents = [tokenize(sentence) for sentence in sentences]

    documents_word_counts = [calculate_word_frequencies(doc) for doc in documents]

    idf_dict = calculate_idf(documents_word_counts)

    tfidfs = []
    for doc, doc_word_counts in zip(documents, documents_word_counts):
        tf_dict = calculate_tf(doc_word_counts, len(doc))
        tfidf_dict = calculate_tfidf(tf_dict, idf_dict)
        tfidfs.append(tfidf_dict)

    tfidf_matrix = pd.DataFrame(tfidfs, index=["Document A", "Document B"]).T
    visualize_tfidf(tfidf_matrix)

    # scikit-learn
    titles = ["seneca", "steve_jobs"]

    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(sentences)
    dict(zip(vectorizer.get_feature_names_out(), vector.toarray()[0]))

    tfidf_df = pd.DataFrame(
        vector.toarray(), index=titles, columns=vectorizer.get_feature_names_out()
    )

    visualize_tfidf(tfidf_df.T)


if __name__ == "__main__":
    main()
