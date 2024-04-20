import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_transfer.npy")
# scores_sentence_space_glove_3c = np.load("results/scores_sentence_space_glove_3c.npy")

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

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
scores = np.load("results/scores_glove_3c.npy")
for method_id, method in enumerate(methods):
    ax.plot(gaussian_filter1d(scores[method_id, :, 8], 9), label="%s | Mean BAC: %.3f" % (method, np.mean(scores[method_id, :, 8])), ls=lss[method_id], c=colors[method_id], lw=lws[method_id])

ax.plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 9), c="red", label="Glove Sentence Space | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove[:, 9]))
ax.plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 9), c="blue", label="Glove Sentence Space | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove[:, 9]))

ax.set_xlim(0, 2731)
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("chunks")
ax.set_ylabel("BAC")
ax.set_title("3 classes")

ax.legend(ncol=1, frameon=False, loc="upper center")

plt.tight_layout()
plt.savefig("figures/4_class_number.png", dpi=200)
