# Modelling

import numpy as np
import pandas as pd
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, cross_val_predict
from scipy.stats import randint

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import scikitplot as skplt
from joblib import dump, load


import pretraitement as prepa
import fonctions as f
    
    
class RandomForest:

    def __init__(self, n_trees, min_samples_split, max_depth, n_classes= 2, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.n_classes= n_classes
        self.trees = []


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
          
            X_samp, y_samp = f.bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)         


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [f.most_common_label(tree_pred) for tree_pred in tree_preds]

        return np.array(y_pred)
    
    
    
    def predict_proba(self, X):
        
        prob= np.array([tree.predict_proba(X) for tree in self.trees])
        
        proba= prob[0]
        
        for p in prob[1:]:
            
            res = proba + p
            proba= res
            
        proba /= self.n_trees
            
        
        return proba
    
