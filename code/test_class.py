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
import pickle


import pretraitement as prepa
import apprentissage as app




class ModelPredictor:
    
    
    
    def __init__(self, modele):
        
        self.model= pickle.load(open(modele, 'rb'))
        
        
    def predict(self, X):
        

        return self.model.predict(X)
    
 
    
    def predict_proba(self, X):
  
        
        return self.model.predict_proba(X)
        