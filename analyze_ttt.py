import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
chunk_size = 250
# n_chunks = ceil(stream.shape[0]/chunk_size)
n_chunks = 2727

score_files = ["tfidf", "MiniLM", "glove_pca100", "glove_nopca"]
titles = ["TF-IDF | max_features=100, ngram_range=(1,2)", "MiniLM | PCA to 100 components", "GloVe | PCA to 100 components", "GloVe | 300 features without PCA"]

for file_id, file in enumerate(score_files):

    scores = np.load("results/scores_%s_2c.npy" % file)
    scores_sentence_space = np.load("results/scores_sentence_space_2c.npy")
    scores_sentence_space_transfer = np.load("results/scores_sentence_space_2c_transfer.npy")
    scores_sentence_space_glove = np.load("results/scores_sentence_space_glove.npy")
    scores_sentence_space_w2v_own = np.load("results/scores_sentence_space_w2v_own.npy")
    scores_sentence_space_w2v_pre = np.load("results/scores_sentence_space_w2v_pre.npy")
    

    metrics=["recall", "precision", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac"]

    methods = [
            "HF",
            "CDS",
            "NIE",
            "KUE",
            "ROSE",
        ]

    colors = ['silver', 'darkorange', 'seagreen', 'darkorchid', 'dodgerblue', 'red', 'blue']
    lws = [1.5, 1.5, 1.5 ,1.5 ,1.5 ,1.5 ,1.5]
    lss = ["-", "-", "-", "-", "-", "-", "-"]

    fig, ax = plt.subplots(1, 1, figsize=(13, 10))

    for method_id, method in enumerate(methods):
        ax.plot(gaussian_filter1d(scores[method_id, :, 8], 9), label=method, ls=lss[method_id], c=colors[method_id], lw=lws[method_id])

    ax.plot(gaussian_filter1d(scores_sentence_space[:, 8], 9), label="SS-MiniLM: %.3f" % np.mean(scores_sentence_space[:, 8]), ls=lss[5], c=colors[5], lw=lws[5])
    ax.plot(gaussian_filter1d(scores_sentence_space_transfer[:, 8], 9), label="SS-MiniLM Transfer: %.3f" % np.mean(scores_sentence_space_transfer[:, 8]), ls="-", c="green", lw=2)
    ax.plot(gaussian_filter1d(scores_sentence_space_glove[:, 8], 9), label="SS-Glove: %.3f" % np.mean(scores_sentence_space_glove[:, 8]), ls=lss[6], c=colors[6], lw=lws[6])
    # ax.plot(gaussian_filter1d(scores_sentence_space_w2v_own[:, 8], 9), label="SS-W2V-OWN: %.3f" % np.mean(scores_sentence_space_w2v_own[:, 8]), ls=lss[4], c=colors[4], lw=lws[4])
    # ax.plot(gaussian_filter1d(scores_sentence_space_w2v_pre[:, 8], 9), label="SS-W2V-PRE: %.3f" % np.mean(scores_sentence_space_w2v_pre[:, 8]), ls=lss[3], c=colors[3], lw=lws[3])
    ax.set_xlim(0, n_chunks)
    ax.grid(ls=":", c=(0.7, 0.7, 0.7))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("chunks")
    ax.set_ylabel("BAC")
    ax.set_title(titles[file_id])

    ax.legend(ncol=3, frameon=False, loc="upper center")

    plt.tight_layout()
    plt.savefig("figures/%s_.png" % file)
    # plt.savefig("figures/sentspace.png")