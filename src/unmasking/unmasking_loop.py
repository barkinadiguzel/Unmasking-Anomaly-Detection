import numpy as np

def unmasking(X, y, num_loops=10, top_features=50):
    from sklearn.metrics import accuracy_score
    from .classifier import train_classifier
    
    acc_profile = []
    X_curr = X.copy()
    
    for _ in range(num_loops):
        clf = train_classifier(X_curr, y)
        y_pred = clf.predict(X_curr)
        acc = accuracy_score(y, y_pred)
        acc_profile.append(acc)
        
        coef_abs = np.abs(clf.coef_).flatten()
        if len(coef_abs) <= top_features:
            break
        top_idx = np.argsort(coef_abs)[-top_features:]
        X_curr = np.delete(X_curr, top_idx, axis=1)
        
    return acc_profile
