import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, cross_val_predict
from scipy.stats import randint
from sklearn.metrics import precision_score

import scikitplot as skplt
from joblib import dump, load
import pickle


import pretraitement as prepa
import fonctions as f
import apprentissage as app
import test_class as te




def prediction (seuil, predictions, matrice, modele, x_test, y):
    
    
    # load the model from disk
    
    model= te.ModelPredictor(modele)
    
    pred= model.predict(x_test)
    
    acc= accuracy_score(y, pred)
    print('\n')
    print("Accuracy:", acc)
    print('\n')
    
    probas = model.predict_proba(x_test)

    
    prediction_new= []

    for i in range(len(probas)):

        x= probas[i][0]

        if x>= seuil: 
            prediction_new.append(0.0)

        else: 
            prediction_new.append(1.0)
            

    matrix= confusion_matrix(y, prediction_new)
    
    
    print("Seuil:", seuil)
    print('\n')
    print("=== Confusion Matrix ===")
    print(matrix)
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y, prediction_new))
    
    p = pd.DataFrame(matrix, columns=['0', '1'])
    p.to_csv(matrice)
    
    
    f= pd.DataFrame(columns=['initial_target', 'predicted_target'], index=range(len(y)))
    f['initial_target']= y.values.reshape(len(y))
    f['predicted_target']= np.array(prediction_new)
    f.to_csv(predictions)

    