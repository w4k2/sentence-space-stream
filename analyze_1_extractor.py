import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d

# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_imgfixed.npy")
scores_sentence_space_minilm = np.load("results/scores_sentence_space_2c_transfer.npy")
scores_sentence_space_w2v_own = np.load("results/scores_sentence_space_w2v_own_imgfixed.npy")
scores_sentence_space_w2v_pre = np.load("results/scores_sentence_space_w2v_pre_imgfixed.npy")

fig, ax = plt.subplots(1, 1, figsize=(8, 8/1.618))

labels = [
    "[~%.3f] GSS" % np.mean(scores_sentence_space_glove[:, 9]),
    "[~%.3f] MiniLM" % np.mean(scores_sentence_space_minilm[:, 9]),
    "[~%.3f] W2V Pretrained" % np.mean(scores_sentence_space_w2v_pre[:, 9]),
    "[~%.3f] W2V Partial Fit" % np.mean(scores_sentence_space_w2v_own[:, 9])
]

ax.plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 20), c="red", label=labels[0])
ax.plot(gaussian_filter1d(scores_sentence_space_minilm[:, 9], 20), c="limegreen", label=labels[1])
ax.plot(gaussian_filter1d(scores_sentence_space_w2v_pre[:, 9], 20), c="blue", label=labels[2])
ax.plot(gaussian_filter1d(scores_sentence_space_w2v_own[:, 9], 20), c="blue", label=labels[3], ls="--")
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("number of processed chunks")
ax.set_ylabel("balanced accuracy score")
ax.set_ylim(.5, 1.)
ax.set_xlim(0, 2500)
ax.set_title("Review of extraction methods")
ax.legend(frameon=False, ncol=1)

plt.tight_layout()
plt.savefig("figures/1_extractor.png", dpi=200)
plt.savefig('foo.png')