import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

scores_glove_nopca = np.load("results/scores_glove_nopca_2c.npy")
scores_glove_pca = np.load("results/scores_glove_pca100_2c.npy")

#fig, ax = plt.subplots(1, 1, figsize=(8, 8/1.618))

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

fig, ax = plt.subplots(1, 2, figsize=(8, 8/1.618))

for method_id, method in enumerate(methods):
    ax[0].plot(gaussian_filter1d(scores_glove_nopca[method_id, :, 8], 9), label="[~%.3f] %s" % (np.mean(scores_glove_nopca[method_id, :, 8]), method), 
               ls=lss[method_id], c=colors[method_id], lw=lws[method_id])
    
    ax[1].plot(gaussian_filter1d(scores_glove_pca[method_id, :, 8], 9), label="[~%.3f] %s" % (np.mean(scores_glove_pca[method_id, :, 8]), method), 
               ls=lss[method_id], c=colors[method_id], lw=lws[method_id])


ax[0].set_xlim(0, 2500)
ax[0].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylim(0.5, 1.0)
ax[0].set_xlabel("chunks")
ax[0].set_ylabel("BAC")
ax[0].set_title("Glove without PCA")

ax[1].set_xlim(0, 2500)
ax[1].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylim(0.5, 1.0)
ax[1].set_xlabel("chunks")
ax[1].set_ylabel("BAC")
ax[1].set_title("Glove with PCA to 100 features")

ax[0].legend(ncol=1, frameon=False, loc="upper right")
ax[1].legend(ncol=1, frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig("figures/2_ref_pca.png", dpi=200)
