import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List

stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens: List[str]) -> List[str]:
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def tokenization_for_review(review: str) -> List[str]:
    tokens = word_tokenize(review)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

def tokenization_and_remove_stopwords(review: str) -> str:
    tokens = tokenization_for_review(review)
    tokens = remove_stopwords(tokens)

    return " ".join(tokens)

if __name__ == '__main__':
    nltk.download("punkt_tab")
    nltk.download("stopwords")
