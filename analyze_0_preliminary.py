import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


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
print(bins[:50])
print(counts[:50])
print(np.sum(counts[:50]))
print(np.sum(counts))
print(np.sum(counts[:50])/np.sum(counts))
print(stream.shape)

fig, ax = plt.subplots(1, 3, figsize=(30, 10))

ax[2].plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 18), c="red", label="Glove Sentence Space Transfer         | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove[:, 9]))
ax[2].plot(gaussian_filter1d(scores_sentence_space_glove_notransfer[:, 9], 18), c="blue", label="Glove Sentence Space Without Transfer | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove_notransfer[:, 9]))
ax[2].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_xlabel("chunks")
ax[2].set_ylabel("BAC")
ax[2].set_ylim(.75, .9)
ax[2].set_xlim(0, 2731)
ax[2].set_title("Trasfer learning preliminary")
ax[2].legend(frameon=False, ncol=1)

ax[0].bar(bins, counts, color="k")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_xlabel("#words")
ax[0].set_ylabel("Counts")
ax[0].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[0].set_yscale("log")
ax[0].set_xticks([i for i in range(0, 110, 10)], [str(i) for i in range(0, 100, 10)] + ["..."])
ax[0].set_xlim(0, 100)
ax[0].vlines(50, 0, 100000, color="red", lw=4)
ax[0].text(47, 250, "~0.9971% of the dataset", c="red", rotation=90, fontsize=20)
ax[0].set_title("Starting image height")

ax[1].plot(gaussian_filter1d(scores_sentence_space_glove_h50[:, 9], 18), c="blue", label="Glove Sentence Space 300x50  | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove_h50[:, 9]))
ax[1].plot(gaussian_filter1d(scores_sentence_space_glove_h100[:, 9], 18), c="limegreen", label="Glove Sentence Space 300x100 | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove_h100[:, 9]))
ax[1].plot(gaussian_filter1d(scores_sentence_space_glove[:, 9], 18), c="red", label="Glove Sentence Space 300x200 | Mean BAC: %.3f" % np.mean(scores_sentence_space_glove[:, 9]))
ax[1].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_xlabel("chunks")
ax[1].set_ylabel("BAC")
ax[1].set_ylim(.75, .9)
ax[1].set_xlim(0, 2731)
ax[1].set_title("Image height preliminary")

ax[1].legend(frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("figures/0_preliminary.png", dpi=200)
