import numpy as np 
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def best_parameters(X_train, y_train):
    
    param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                     param_distributions = param_dist, 
                                     n_iter=5, 
                                     cv=10)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_
    
    param=  rand_search.best_params_
    depth= param['max_depth']
    estimators= param['n_estimators']
    
    return depth, estimators
    
    
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[idxs], y.iloc[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common




