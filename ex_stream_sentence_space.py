import numpy as np
from math import ceil
from skmultiflow.trees import HoeffdingTree
from strlearn.metrics import balanced_accuracy_score as bac
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from strlearn.ensembles import KUE, ROSE, NIE
from utils import CDS
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
y = np.load("fakeddit_stream/fakeddit_posts_y.npy")
print(X.shape)
print(y.shape)

# Only titles, without timestamp
# Binary problem
stream = X[:, 0]
y_2 = y[:, 0]
y_2[y_2==0] = -1
y_2[y_2==1] = 0
y_2[y_2==-1] = 1

chunk_size = 250
# All chunks
n_chunks = ceil(stream.shape[0]/chunk_size)
# To always have both classes
n_chunks = 2727
n_estimators = 10

metrics=(recall, recall_score, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac, balanced_accuracy_score)
# clf = HoeffdingTree(split_criterion="hellinger")
# clf = GaussianNB()
methods = [
        HoeffdingTree(split_criterion="hellinger"),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ]

# METHODS x CHUNKS x METRICS
scores = np.zeros((5, n_chunks, 10))
for chunk_id in tqdm(range(n_chunks)):
    chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    chunk_y = y_2[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    
    print("EMBEDDINGS")
    embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2', device="mps").to("mps")
    print(model.device)
    for text_id, text in enumerate(tqdm(chunk_X[26:])):
        words = text.split(" ")
        if len(words) != 1:
            embeddings.append(model.encode(words))
            
        plt.imshow(model.encode(words))
        plt.savefig("bar.png")
        exit()
            
            

    embeddings = np.array(embeddings, dtype=object)
    print(embeddings[0], embeddings[0].shape)
    
    exit()
    
    for method_id, method in enumerate(methods):
        if chunk_id == 0:
            method.fit(preproc_X, chunk_y)
        else:
            pred = method.predict(preproc_X)
            
            for metric_id, metric in enumerate(metrics):
                score = metric(chunk_y, pred)
                scores[method_id, chunk_id, metric_id] = score
            
            method.partial_fit(preproc_X, chunk_y)

np.save("results/scores_embeddings", scores)