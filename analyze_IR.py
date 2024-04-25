import numpy as np
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
bias = np.load("fakeddit_stream/fakeddit_posts_y.npy")
# How many classes?
bias_id = 0
print(X.shape)
print(bias.shape)

# Only titles, without timestamp
# Binary problem
stream = X[:, 0]
y = np.array([0,1])[bias[:,bias_id]] if bias_id == 0 else bias[:,bias_id]

chunk_size = 250
# All chunks
n_chunks = ceil(stream.shape[0]/chunk_size)
# Select dummies
classes = np.unique(y)
n_classes = len(classes)
dummies = stream[[np.where(y==label)[0][0] for label in classes]]

irs = np.zeros((2, n_chunks))
for chunk_id in tqdm(range(n_chunks)):
    chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    chunk_y = y[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    
    if len(np.unique(chunk_y)) != n_classes:
        chunk_X[:n_classes] = dummies
        chunk_y[:n_classes] = classes
        
    _, counts = np.unique(chunk_y, return_counts=True)
    
    sum = np.sum(counts)
    irs[0, chunk_id] = counts[0]/sum
    irs[1, chunk_id] = counts[1]/sum


fig, ax = plt.subplots(1, 1, figsize=(10, 5))


ax.plot(gaussian_filter1d(irs[0], 3), label="True", c="blue")
ax.plot(gaussian_filter1d(irs[1], 3), label="Fake", c="red")
ax.set_xlabel("chunks")
ax.set_ylabel("Prior class probability")
ax.legend(frameon=False)
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures/prior.png", dpi=200)