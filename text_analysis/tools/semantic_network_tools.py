from collections import Counter
from itertools import combinations
import networkx as nx
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm

def symmetrize_co_occr_matrix(co_occr: dict = None, min_freq: int = 5) -> dict:
    co_symmetrize = Counter()
    for (w1, w2), f in co_occr.items():
        key = tuple(sorted([w1, w2]))
        co_symmetrize[key] += f

    return co_symmetrize

def gen_co_occr_matrix(window_size: int = 5, review_list: List[str] = None) -> dict:
    co_occurrence = Counter()
    lemmatizer = WordNetLemmatizer()

    for review in tqdm(review_list, desc='generate cooccurrence matrix'):
        tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(review)]
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:(i+window_size)]
            for w1, w2 in combinations(window, 2):
                if w1 != w2:
                    co_occurrence[(w1, w2)] += 1

    return co_occurrence

def plot_co_occr_network(window_size: int = 5, review_list: List[str] = None, min_freq: int = 10, label: str = "") -> None:
    co_occurrence = gen_co_occr_matrix(window_size=window_size, review_list=review_list)
    co_symmetrize = symmetrize_co_occr_matrix(co_occurrence)
    print(f"symmetrize ended!")
    G = nx.Graph()
    for (w1, w2), f in co_symmetrize.items():
        if f >= min_freq:
            G.add_edge(w1, w2, weight=f)
    print(f"start to plot!")
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5)
    deg = nx.degree_centrality(G)
    nx.draw_networkx_nodes(G, pos, node_size=[v*800 for v in deg.values()], node_color='skyblue')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(f"{label} - Co-occurrence Semantic Network" if label else "")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"plot ended!")

if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw-1.4') 
    nltk.download('punkt')  
