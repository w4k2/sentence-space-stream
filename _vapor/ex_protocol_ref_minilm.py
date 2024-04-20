import numpy as np
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
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
from sklearn.decomposition import PCA
import os
from sklearn.base import clone

# Configure processing
experiment_name = 'ref_minilm'
bias_id = 0
chunk_size = 250    # need to be dividable by 10
n_splits = 2        # do not change me
n_repeats = 5       # do not change me
n_folds = n_splits * n_repeats

# Replace with real methods and metrics but preserve the counts
n_estimators = 10
methods = [
        HoeffdingTree(split_criterion="hellinger"),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ]

metrics=(recall, recall_score, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac, balanced_accuracy_score)

n_methods = len(methods)
n_metrics = len(metrics)

# Load data
# data = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
X = np.load("results/embeddings_ref_minilm.npy")
bias = np.load("fakeddit_stream/fakeddit_posts_y.npy")

# Select processing context
# X = data[:, 0]
y = np.array([1,0])[bias[:,bias_id]] if bias_id == 0 else bias[:,bias_id]

n_chunks = X.shape[0] // chunk_size
limit = n_chunks * chunk_size

# X = X[:limit]
y = y[:limit]

print('%i chunks of size %i' % (n_chunks, chunk_size))

# Preliminary analysis
classes = np.unique(y)
n_classes = len(classes)

# Select dummies
dummies = X[[np.where(y==label)[0][0] for label in classes]]

# Prepare the crossval
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1410)
trains = []
tests = []
for train, test in tqdm(rskf.split(X, y)):
    trains.append(train)
    tests.append(test)
    
trains = np.array(trains)   # fold X idx
tests = np.array(tests)     # fold X idx

# Initialize the result tables
stationary_results = np.zeros((n_folds, n_methods, n_metrics))
streaming_results = np.zeros((n_folds, n_chunks//n_splits, n_methods, n_metrics))

# model = SentenceTransformer('all-MiniLM-L6-v2', device="mps").to("mps")
pca_stationary = PCA(100, random_state=1410)
pca_stream = PCA(100, random_state=1410)

"""
Do the STATIONARY modeling here
"""
# print("| ENCODING")
# embeddings = model.encode(X)
# np.save("results/embeddings.npy", embeddings)
# exit()

print('| STATIONARY')
for fold in tqdm(range(n_folds)):
    print('Fold %i' % fold)
    
    # Stationary evaluation
    train, test = trains[fold], tests[fold]
    
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    
    # X_train_stat = pca_stationary.fit_transform(X_train)
    # X_test_stat = pca_stationary.transform(X_test)
    
    for clf_idx, clf in enumerate(methods):
        clf = clone(clf) # For proper methods use clone
        
        # clf.fit(X_train_stat, y_train)
        
        for metric_idx, metric in enumerate(metrics):
            # y_pred = clf.predict(X_test_stat)
            # score = metric(y_test, y_pred)

            stationary_results[fold, clf_idx, metric_idx] = .5 # use score instead of .5
    
    # Streaming evaluation
    print('| STREAMING')
    
    # Initialize
    clfs = [clone(clf) for clf in methods] # For proper methods use clone

    # Do the chunks
    for chunk_id in tqdm(range(n_chunks//n_splits)):
        # Gather separate training-testing chunks
        start, stop = chunk_id * chunk_size, (chunk_id + 1) * chunk_size
        s_X_train, s_y_train = X_train[start:stop], y_train[start:stop]
        s_X_test, s_y_test = X_test[start:stop], y_test[start:stop]
        
        if chunk_id == 0:
            s_X_train = pca_stream.fit_transform(s_X_train)
            s_X_test = pca_stream.transform(s_X_test)
        else:
            s_X_train = pca_stream.transform(s_X_train)
            s_X_test = pca_stream.transform(s_X_test)
        
        # Dummy validation
        if len(np.unique(s_y_train)) != n_classes:
            s_X_train[:n_classes] = pca_stream.transform(dummies)
            s_y_train[:n_classes] = classes    
            
        # Train
        for clf_idx, clf in enumerate(methods):
            clf.partial_fit(s_X_train, s_y_train)
            
        # Test
        for clf_idx, clf in enumerate(methods):
            y_pred = clf.predict(s_X_test)
        
            for metric_idx, metric in enumerate(metrics):
                score = metric(s_y_test, y_pred)
                
                streaming_results[fold, chunk_id, clf_idx, metric_idx] = score
    
# np.save('results/%s-stationary.npy' % experiment_name, stationary_results)
np.save('results/%s-streaming.npy' % experiment_name, streaming_results)