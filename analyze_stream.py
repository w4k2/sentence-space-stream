import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

score_files = ["tfidf", "MiniLM"]
titles = ["TF-IDF | max_features=100, ngram_range=(1,2)", "EMBEDDINGS | PCA to 100 components"]

for file_id, file in enumerate(score_files):

    scores = np.load("results/scores_%s.npy" % file)
    scores_sentence_space = np.load("results/scores_sentence_space.npy")

    n_chunks = 2727

    metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

    methods = [
            "HF",
            "CDS",
            "NIE",
            "KUE",
            "ROSE",
        ]

    colors = ['silver', 'darkorange', 'seagreen', 'darkorchid', 'dodgerblue', 'red']
    lws = [1.5, 1.5, 1.5 ,1.5 ,1.5 ,2]
    lss = ["-", "-", "-", "-", "-", "-"]

    fig, ax = plt.subplots(1, 1, figsize=(13, 10))

    for method_id, method in enumerate(methods):
        ax.plot(gaussian_filter1d(scores[method_id, :, 8], 9), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])

    ax.plot(gaussian_filter1d(scores_sentence_space[:, 8], 9), label="SS", ls=lss[5], c=colors[5], lw=lws[5])
    ax.set_xlim(0, n_chunks)
    ax.grid(ls=":", c=(0.7, 0.7, 0.7))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("chunks")
    ax.set_ylabel("BAC")
    ax.set_title(titles[file_id])

    ax.legend(ncol=6, frameon=False, loc="upper center")

    plt.tight_layout()
    plt.savefig("figures/%s.png" % file)