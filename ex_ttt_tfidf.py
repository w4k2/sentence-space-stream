import numpy as np
from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultiflow.trees import HoeffdingTree
from strlearn.metrics import balanced_accuracy_score as bac
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from strlearn.ensembles import KUE, ROSE, NIE
from utils import CDS
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score
from sklearn.decomposition import PCA


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

chunk_size = 250
# All chunks
n_chunks = ceil(stream.shape[0]/chunk_size)
# Select dummies
classes = np.unique(y)
n_classes = len(classes)
dummies = stream[[np.where(y==label)[0][0] for label in classes]]

metrics=(recall, recall_score, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac, balanced_accuracy_score)

n_estimators = 10
methods = [
        HoeffdingTree(split_criterion="hellinger"),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ]

# METHODS x CHUNKS x METRICS
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
tfidf.fit(stream, y)

scores = np.zeros((5, n_chunks, 10))
for chunk_id in tqdm(range(n_chunks)):
    chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    chunk_y = y[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    
    if len(np.unique(chunk_y)) != n_classes:
        chunk_X[:n_classes] = dummies
        chunk_y[:n_classes] = classes
    
    preproc_X = tfidf.transform(chunk_X).A
    
    for method_id, method in enumerate(methods):
        if chunk_id == 0:
            method.fit(preproc_X, chunk_y)
        else:
            pred = method.predict(preproc_X)
            
            for metric_id, metric in enumerate(metrics):
                score = metric(chunk_y, pred)
                scores[method_id, chunk_id, metric_id] = score
            
            method.partial_fit(preproc_X, chunk_y)

np.save("results/scores_tfidf", scores)