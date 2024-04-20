import numpy as np
from math import ceil
from skmultiflow.trees import HoeffdingTree
from tqdm import tqdm
from strlearn.ensembles import KUE, ROSE, NIE
from utils import CDS
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import os
from time import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"


X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
bias = np.load("fakeddit_stream/fakeddit_posts_y.npy")
# How many classes?
bias_id = 0
print(X.shape)
print(bias.shape)

# Only titles, without timestamp
# Binary problem
stream = X[:, 0]
y = np.array([1,0])[bias[:,bias_id]] if bias_id == 0 else bias[:,bias_id]
print(np.unique(y, return_counts=True))

chunk_size = 250
# All chunks
n_chunks = 110
# Select dummies
classes = np.unique(y)
n_classes = len(classes)
dummies = stream[[np.where(y==label)[0][0] for label in classes]]

n_estimators = 10

model = SentenceTransformer('all-MiniLM-L6-v2', device="mps").to("mps")

# METHODS x CHUNKS x METRICS
pca = PCA(100, random_state=1410)
# REPLICATION x METHODS x CHUNKS x (EXTRACTION, TEST, TRAIN)
times = np.zeros((10, 5, n_chunks, 3))
for replication_id in range(10):
    methods = [
        HoeffdingTree(split_criterion="hellinger"),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ]
    
    for chunk_id in tqdm(range(n_chunks)):
        chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
        chunk_y = y[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
        
        if len(np.unique(chunk_y)) != n_classes:
            chunk_X[:n_classes] = dummies
            chunk_y[:n_classes] = classes
        
        start_extraction = time()
        print(chunk_X)
        embeddings = model.encode(chunk_X)
        print(embeddings)
        stop_extraction = time()
        
        for method_id, method in enumerate(methods):
            if chunk_id == 0:
                preproc_X = pca.fit_transform(embeddings)
                method.fit(preproc_X, chunk_y)
            else:
                start_prediction = time()
                preproc_X = pca.transform(embeddings)
                pred = method.predict(preproc_X)
                stop_prediction = time()
                start_training = time()
                method.partial_fit(preproc_X, chunk_y)
                stop_training = time()
                
                times[replication_id, method_id, chunk_id, 0] = stop_extraction-start_extraction
                times[replication_id, method_id, chunk_id, 1] = stop_prediction-start_prediction
                times[replication_id, method_id, chunk_id, 2] = stop_training-start_training
            
# np.save("results/time_complexity_ref", times)