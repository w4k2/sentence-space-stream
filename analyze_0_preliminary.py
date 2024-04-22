import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
stream = X[:, 0]

scores_sentence_space_glove_h50 = np.load("results/scores_sentence_space_glove_imgfixed_50.npy")
scores_sentence_space_glove_h100 = np.load("results/scores_sentence_space_glove_imgfixed_100.npy")
scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_imgfixed.npy")
scores_sentence_space_glove_notransfer = np.load("results/scores_sentence_space_glove_imgfixed_200_notransfer.npy")

lengths = []
for i in tqdm(stream):
    words = i.split(" ")
    lengths.append(len(words))
lengths = np.array(lengths)

# counts, bins = np.histogram(lengths, bins=3)
bins, counts = np.unique(lengths, return_counts=True)
counts = counts.astype(np.float32)
counts /= np.sum(counts)

fig, ax = plt.subplots(1, 3, figsize=(15, 10/1.618))

labels = [
    "[~%.3f] GSS Transfer" % np.mean(scores_sentence_space_glove[:, 9]),
    "[~%.3f] GSS Without Transfer" % np.mean(scores_sentence_space_glove_notransfer[:, 9])
]
ax[2].plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 20, mode='wrap')[:-50], c="red", label=labels[0])
ax[2].plot(gaussian_filter1d(scores_sentence_space_glove_notransfer[:, 9], 20, mode='wrap')[:-50], c="black", label=labels[1])
ax[2].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_xlabel("number of processed chunks")
ax[2].set_ylabel("balanced accuracy score")
ax[2].set_ylim(.5, 1.)
ax[2].set_xlim(0, 2500)
ax[2].set_title("Trasfer learning ablation")
ax[2].legend(frameon=False, ncol=1)

# ax[0].bar(bins, counts, color="k")
ax[0].plot(bins, counts, color='black')
ax[0].fill_between(bins[:50], counts[:50], counts[:50]*0, color='red')
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_xlabel("number of words")
ax[0].set_ylabel("probability [log]")
ax[0].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[0].set_yscale("log")
ax[0].set_xticks([i for i in range(0, 110, 10)], [str(i) for i in range(0, 100, 10)] + ["..."])
ax[0].set_xlim(1, 100)
ax[0].set_ylim(1e-6, 1e0)
ax[0].vlines(50, 0, 1e5, color="red", lw=2)
ax[0].text(47, .01, "~0.9971% of the dataset", c="red", rotation=90, fontsize=12, ha='center', va='center')
ax[0].set_title("Image size distribution")

labels = [
    "[~%.3f] GSS 300x50" % np.mean(scores_sentence_space_glove_h50[:, 9]),
    "[~%.3f] GSS 300x100" % np.mean(scores_sentence_space_glove_h100[:, 9]),
    "[~%.3f] GSS 300x200" % np.mean(scores_sentence_space_glove[:, 9])
]

ax[1].plot(gaussian_filter1d(scores_sentence_space_glove_h50[:, 9], 20), c="blue", label=labels[0])
ax[1].plot(gaussian_filter1d(scores_sentence_space_glove_h100[:, 9], 20), c="limegreen", label=labels[1])
ax[1].plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 20), c="red", label=labels[2])
ax[1].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_xlabel("number of processed chunks")
ax[1].set_ylabel("balanced accuracy score")
ax[1].set_ylim(.5, 1.)
ax[1].set_xlim(0, 2500)
ax[1].set_title("Preliminary review")
ax[1].legend(frameon=False, ncol=1)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("figures/0_preliminary.png", dpi=200)
