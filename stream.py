import numpy as np
from math import ceil
X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
y = np.load("fakeddit_stream/fakeddit_posts_y.npy")
print(X.shape)
print(y.shape)

# Only titles, without timestamp
# Binary problem
stream = X[:, 0]
y_2 = y[:, 0]

chunk_size = 250
# All chunks
n_chunks = ceil(stream.shape[0]/chunk_size)
# To always have both classes
n_chunks = 2727

for chunk_id in range(n_chunks):
    chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    chunk_y = y_2[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    print(chunk_X)
    print(chunk_y)

    # Do we have both classes?
    # labels, counts = np.unique(chunk_y, return_counts=True)
    # if counts.shape[0] < 2:
    #     print(counts)
    #     print(chunk_id)
    #     exit()