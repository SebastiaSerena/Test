# Modelling

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, cross_val_predict
from scipy.stats import randint
from sklearn.metrics import precision_score


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import scikitplot as skplt
from joblib import dump, load
import pickle


import pretraitement as prepa
import fonctions as f
import training_class as tr



def apprentissage (max_depth, n_trees, min_samples_split, seuil, X_train, X_test, y_train, y_test, modele):
    
    
    model= tr.RandomForest(n_trees, min_samples_split, max_depth)
    
    model.fit(X_train, y_train)
    
    pred= model.predict(X_test)
    
    acc= accuracy_score(y_test, pred)
    print('\n')
    print("Accuracy:", acc)
    print('\n')
    
    probas = model.predict_proba(X_test)

    
    prediction_new= []

    for i in range(len(probas)):

        x= probas[i][0]

        if x>= seuil: 
            prediction_new.append(0.0)

        else: 
            prediction_new.append(1.0)
            

    matrix= confusion_matrix(y_test, prediction_new)
    
    
    print("Seuil:", seuil)
    print('\n')
    print("=== Confusion Matrix ===")
    print(matrix)
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, prediction_new))
    

    #print('precision score:' precision_score(y_test, pred, average=None))
    
    
    ##dump the model into a file
    with open(modele, 'wb') as f_out:
        pickle.dump(model, f_out) # write final_model in .bin file
        f_out.close()  # close the file 
    
    
    
    return model
    