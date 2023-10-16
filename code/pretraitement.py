import logging
import os.path
from os import path

import sys
#from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import fonctions as f


logger = logging.getLogger(__name__)

# ===================================================================================================


def prepare_data(config):

    # Constantes depuis le fichier de config
    path = config["DATA"]["FILE_PATH"]
    
    train_data = config["DATA"]["TRAIN_DATA"]
    
    test_data = config["DATA"]["TEST_DATA"]
    
    test_size = config["MODELE"]["TEST_SIZE"]
        
    seuil = config["MODELE"]["SEUIL"]
    
    max_depth = config["MODELE"]["DEPH"]
    
    n_trees = config["MODELE"]["TREES"]
    
    min_samples_split= config["MODELE"]["MIN_SPLIT"]
    
    modele = config["MODELE"]["MODELE"]

    predictions = config["OUTPUT"]["PREDICTIONS"]
    
    matrice = config["OUTPUT"]["MATRICE"]
    
 
    logger.info(f'Chargement des données')
    
    try:

        train= pd.read_csv(train_data)
        test= pd.read_csv(test_data)


    except:
        
        df= pd.read_csv(path, sep=";")

        logger.info(f'Exploration des données')

        #print(df.info())
        #print(df.isna().sum())

        df.drop(df.tail(2).index,inplace = True)

        logger.info(f'Nettoyage des données')

        df= df.drop(columns=["id_client"])

        df["annual_flux"].fillna('0', inplace = True)
        df["job_category"].fillna('inconnu', inplace = True) 
        df["risk_of_credit"].fillna('inconnu', inplace = True) 
        df['annual_flux']= df['annual_flux'].str.replace(',', '.', regex=True)

        #get all categorical columns
        cat_columns = ['job_category','service_channel','risk_of_credit','credit_flux']

        #convert all categorical columns to numeric
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

        df = df.astype(float, errors = 'raise')

        train, test = train_test_split(df, test_size=test_size, random_state=42)

        train.to_csv(train_data)
        test.to_csv(test_data)
        
    
    X= train.drop(['target'],axis=1)
    Y= train["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    
    x_test= test.drop(['target'],axis=1)
    y= test["target"]
    
    '''
    
    logger.info(f'Choix des meilleurs paramètres pour le modèle')
    
    max_depth, n_trees= f.best_parameters(X_train, y_train)
    
    print('\n')
    print("max_depth:", max_depth)
    print('\n')
    print("n_trees:", n_trees)
    '''
    
    return max_depth, n_trees, min_samples_split, X_train, X_test, y_train, y_test, x_test, y, seuil, modele, predictions, matrice


        
        
        
    
    
    
    
    
    
    