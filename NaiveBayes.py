import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Tuple
import math

class NaiveBayes:
    def __init__(self, alpha = 1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = None
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

    def fit(self, X: List[str], y: List[int]):
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_tfidf.shape[1]

        # Calculate class priors
        class_count = np.bincount(y)
        self.class_log_prior_ = {c: math.log(count / len(y)) for c, count in zip(self.classes_, class_count)}

        # Calculate feature probabilities
        for c in self.classes_:
            X_c = X_tfidf[y == c]
            N_c = X_c.sum(axis=0) + self.alpha
            # Ensure we get a 1D array
            self.feature_log_prob_[c] = np.log((N_c / N_c.sum()).A1)  # Convert matrix to 1D array


        # Ensure all feature_log_prob_ have the same shape
        expected_shape = (n_features,)
        # Calculate feature probabilities
        for c in self.classes_:
            actual_shape = self.feature_log_prob_[c].shape
            if actual_shape != expected_shape:
                raise ValueError(f"Unexpected shape for class {c}. Expected {expected_shape}, got {self.feature_log_prob_[c].shape}")

    def predict_prob(self, X: List[str]):
        X_tfidf = self.vectorizer.transform(X)
        n_samples = X_tfidf.shape[0]
        n_classes = len(self.classes_)
        log_prob = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes_):
            feature_log_prob = self.feature_log_prob_[c]  # Should be 1D
            if feature_log_prob.shape[0] != X_tfidf.shape[1]:
                raise ValueError(f"Feature dimensions mismatch. X_tfidf: {X_tfidf.shape[1]}, feature_log_prob: {feature_log_prob.shape[0]}")
            log_prob[:, i] = self.class_log_prior_[c] + X_tfidf.dot(feature_log_prob)

        # Normalize probabilities
        log_prob_sum = np.logaddexp.reduce(log_prob, axis=1)
        return np.exp(log_prob - log_prob_sum[:, np.newaxis])

    def predict(self, X: List[str]):
        return self.classes_[np.argmax(self.predict_prob(X), axis= 1)]

def evaluate_model(X: List[str], y: List[int], n_splits= 5):
    y = np.asarray(y).ravel() # ensure y is a 1D array
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    conf_matrix_sum = None

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        model = NaiveBayes()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average = 'weighted'))
        recalls.append(recall_score(y_test, y_pred, average = 'weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average = 'weighted'))

        cm = confusion_matrix(y_test, y_pred)
        if conf_matrix_sum is None:
            conf_matrix_sum = cm
        else:
            conf_matrix_sum += cm

        
    return {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1_scores),
        'confusion_matrix': conf_matrix_sum
    }

def optimize_hyperparameters(X: List[str], y: List[int], alphas = [0.1, 0.5, 1.0, 2.0, 5.0]):
    y = np.asarray(y).ravel()  # ensure y is a 1D array
    best_alpha = None
    best_score = -1

    for alpha in alphas:
        model = NaiveBayes(alpha = alpha)
        scores = evaluate_model(X, y)
        if scores['f1_score'] > best_score:
            best_score = scores['f1_score']
            best_alpha = alpha

    return best_alpha

