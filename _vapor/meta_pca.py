import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
# from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


class MetaPCA(BaseEstimator, ClassifierMixin):
    """
    Sample Weighted Meta Estimator.
    """

    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        self.clf_ = clone(self.base_classifier)
        # self.pca_ = PCA(n_components=100)
        self.pca_ = TfidfVectorizer(max_features=100, ngram_range=(1,2))
        
        X_preproc = self.pca_.fit_transform(X).A
        
        self.clf_.fit(X_preproc, y)
        return self

    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, 'clf_'):
            self.clf_ = clone(self.base_classifier)
            
        if not hasattr(self, 'pca_'):
            self.pca_ = TfidfVectorizer(max_features=100, ngram_range=(1,2))
            print(X.shape)
            X_preproc = self.pca_.fit_transform(X).A
        else:
            X_preproc = self.pca_.transform(X).A

        classes = np.unique(y)
        exit()
        self.clf_.partial_fit(X_preproc, y, classes)

        return self

    def predict_proba(self, X):
        X_preproc = self.pca_.transform(X).A
        return self.clf_.predict_proba(X_preproc)

    def predict(self, X):
        X_preproc = self.pca_.transform(X).A
        return self.clf_.predict(X_preproc)