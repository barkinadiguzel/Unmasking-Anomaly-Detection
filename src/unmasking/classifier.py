from sklearn.linear_model import LogisticRegression

def train_classifier(X, y, C=0.01):
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X, y)
    return clf
