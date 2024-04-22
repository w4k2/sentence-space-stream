import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_imgfixed.npy")

ref_extractors = ["Glove", "W2V", "MiniLM", "TF-IDF"]
ref_extractors_files = ["scores_glove_pca100_2c.npy", "scores_w2v_pre_pca100.npy", "scores_MiniLM_2c.npy", "scores_tfidf_2c.npy"]

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

fig, ax = plt.subplots(2, 2, figsize=(12, 12*.618))
ax = ax.ravel()

for extractor_id, extractor in enumerate(ref_extractors):
    
    scores = np.load("results/%s" % ref_extractors_files[extractor_id])
    print(scores.shape)
    for method_id, method in enumerate(methods):
        ax[extractor_id].plot(gaussian_filter1d(scores[method_id, :, 8], 9), 
                              label="[~%.3f] %s" % (np.mean(scores[method_id, :, 8]), method), ls=lss[method_id], c=colors[method_id], lw=lws[method_id])

    ax[extractor_id].plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 9), c="red", label="[~%.3f] GSS" % np.mean(scores_sentence_space_glove[:, 9]))

    ax[extractor_id].set_xlim(0, 2731)
    ax[extractor_id].grid(ls=":", c=(0.7, 0.7, 0.7))
    ax[extractor_id].spines[['right', 'top']].set_visible(False)
    ax[extractor_id].set_ylim(0.49, 1.0)
    ax[extractor_id].set_xlabel("number of processed chunks")
    ax[extractor_id].set_ylabel("balanced accuracy score")
    ax[extractor_id].set_title(extractor)

    ax[extractor_id].legend(ncol=2, frameon=False, loc="upper right")
    
    # if extractor == "MiniLM":
    #     mlp = np.load("results/scores_MiniLM_MLP.npy")
    #     ax[extractor_id].plot(gaussian_filter1d(mlp[0, :, 9], 9), c="black", label="MLP | Mean BAC: %.3f" % np.mean(mlp[0, :, 9]))
    plt.tight_layout()
    plt.savefig("figures/3_comparison.png", dpi=200)
    plt.savefig('foo.png')
