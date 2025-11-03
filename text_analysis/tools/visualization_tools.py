import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import numpy as np

def word_freq_statistics(all_review: str):
    return Counter(all_review.split())

def draw_word_cloud(review_list: List[str], label: str = "", k: int = 10, max_words: int = 200, mask_path: str = None, colormap: str = 'viridis') -> None:
    all_review = " ".join(review_list)

    # word frequency statistics
    word_freq = word_freq_statistics(all_review)
    top_words = word_freq.most_common(k)
    words, freqs = zip(*top_words)

    plt.figure(figsize=(30, 20))
    plt.subplot(2, 1, 1)
    plt.barh(words, freqs, color="skyblue")
    plt.xlabel("words")
    plt.ylabel("frequency")
    plt.title(f"Top {k} Most Frequent words" + (f"- {label}" if label else ""))

    # drar word clouds
    mask_image = np.array(Image.open(mask_path).convert("L")) if mask_path else None
    word_cloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        max_words=max_words,
        colormap=colormap,
        mask=mask_image
    ).generate(all_review)

    plt.subplot(2, 1, 2)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
