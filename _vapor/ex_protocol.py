import numpy as np
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold

# Configure processing
experiment_name = 'dummy-experiment'
bias_id = 2
chunk_size = 250    # need to be dividable by 10
n_splits = 2        # do not change me
n_repeats = 5       # do not change me
n_folds = n_splits * n_repeats

# Replace with real methods and metrics but preserve the counts
methods = {'A':None, 'B':None, 'C':None, 'D':None}
metrics = {'a':None, 'b':None, 'c':None}

n_methods = len(methods)
n_metrics = len(metrics)

# Load data
data = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
bias = np.load("fakeddit_stream/fakeddit_posts_y.npy")

# Select processing context
X = data[:, 0]
y = np.array([1,0])[bias[:,bias_id]] if bias_id == 0 else bias[:,bias_id]

n_chunks = X.shape[0] // chunk_size
limit = n_chunks * chunk_size

X = X[:limit]
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

for fold in range(n_folds):
    print('Fold %i' % fold)
    
    # Stationary evaluation
    train, test = trains[fold], tests[fold]
    
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    
    for clf_idx, clf_name in enumerate(methods):
        clf = methods[clf_name] # For proper methods use clone
        
        # clf.fit(X_train, y_train)
        
        for metric_idx, metric_name in enumerate(metrics):
            metric = metrics[metric_name]
            
            # y_pred = clf.predict(X_test)
            # score = metric(y_test, y_pred)

            stationary_results[fold, clf_idx, metric_idx] = .5 # use score instead of .5
    
    """
    Do the STATIONARY modeling here
    """
    print('| STATIONARY')
    
    # Streaming evaluation
    print('| STREAMING')
    
    # Initialize
    clfs = [methods[clf_name] for clf_name in methods] # For proper methods use clone

    # Do the chunks
    for chunk_id in tqdm(range(n_chunks//n_splits)):
        # Gather separate training-testing chunks
        start, stop = chunk_id * chunk_size, (chunk_id + 1) * chunk_size
        s_X_train, s_y_train = X_train[start:stop], y_train[start:stop]
        s_X_test, s_y_test = X_test[start:stop], y_test[start:stop]
        
        # Dummy validation
        if len(np.unique(s_y_train)) != n_classes:
            s_X_train[:n_classes] = dummies
            s_y_train[:n_classes] = classes    
            
        # Train
        for clf_idx, clf_name in enumerate(methods):
            clf = clfs[clf_idx]

            # clf.partial_fit(s_X_train, s_y_train)
            
        # Test
        for clf_idx, clf_name in enumerate(methods):
            clf = clfs[clf_idx]

            # y_pred = clf.predict(s_X_test)
        
            for metric_idx, metric_name in enumerate(metrics):
                metric = metrics[metric_name]
                
                # score = metric(s_y_test, y_pred)
                
                streaming_results[fold, chunk_id, clf_idx, metric_idx] = .5 # use score instead of .5
    
np.save('results/%s-stationary.npy' % experiment_name, stationary_results)
np.save('results/%s-streaming.npy' % experiment_name, streaming_results)