import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_imgfixed.npy")
scores_sentence_space_minilm = np.load("results/scores_sentence_space_2c_transfer.npy")
scores_sentence_space_w2v_own = np.load("results/scores_sentence_space_w2v_own_imgfixed.npy")
scores_sentence_space_w2v_pre = np.load("results/scores_sentence_space_w2v_pre_imgfixed.npy")

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

ax.plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 18), c="red", label="Glove Sentence Space           | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove[:, 9]))
ax.plot(gaussian_filter1d(scores_sentence_space_minilm[:, 9], 18), c="limegreen", label="MiniLM Sentence Space          | Mean BAC: %.3f" % np.mean(scores_sentence_space_minilm[:, 9]))
ax.plot(gaussian_filter1d(scores_sentence_space_w2v_pre[:, 9], 18), c="blue", label="W2V Pretrained Sentence Space  | Mean BAC: %.3f" % np.mean(scores_sentence_space_w2v_pre[:, 9]))
ax.plot(gaussian_filter1d(scores_sentence_space_w2v_own[:, 9], 18), c="blue", label="W2V Partial Fit Sentence Space | Mean BAC: %.3f" % np.mean(scores_sentence_space_w2v_own[:2, 9]), ls="--")
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("chunks")
ax.set_ylabel("BAC")
ax.set_ylim(.75, .9)
ax.set_xlim(0, 2731)
ax.set_title("Choosing extraction method")
ax.legend(frameon=False, ncol=1)

plt.tight_layout()
plt.savefig("figures/1_extractor.png", dpi=200)
