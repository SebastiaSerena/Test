import argparse
import logging
from pathlib import Path
import yaml

import pretraitement as prepa
import fonctions as f
import apprentissage as app
import prediction as pred
import training_class as train
import test_class as te


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default=None, help="config file to use", required=True
    )
    parser.add_argument(
        "--display", default=False, help="Print loaded config", action="store_true"
    )
    parser.add_argument(
        "--train", default=False, help="Train the model", action="store_true"
    )
    
    parser.add_argument(
        "--test", default=False, help="Test the model", action="store_true"
    )
    
    #parser.add_argument("--dataset", required=True, help="Path of the dataset")
    parser.add_argument("--output", default="output", help="Directory to ouptut files")
    arguments = parser.parse_args()
    # Create output dir
    p = Path(arguments.output)
    p.mkdir(exist_ok=True, parents=True)

    # Creating logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Create handlers
    # logging to console and file
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f"{arguments.output}/file.log")

    console_handler.setLevel(level=logging.INFO)
    file_handler.setLevel(level=logging.DEBUG)

    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(format)
    file_handler.setFormatter(format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    with open(arguments.config) as fh:
        config = yaml.safe_load(fh)

    if arguments.display:
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config)
        
    max_depth, n_trees, min_samples_split, X_train, X_test, y_train, y_test, x_test, y, seuil, modele, predictions, matrice = prepa.prepare_data(config)
    
    if arguments.train:
        
        logger.info(f"Debut de l'apprentissage")
        
        model= app.apprentissage (max_depth, n_trees, min_samples_split, seuil, X_train, X_test, y_train, y_test, modele)
        
        '''
        model_loaded, score_1 = test(
            config, arguments.output, test_loader
        )
        
        '''
        
        logger.info(f"Fin de l'apprentissage")
        
    
    if arguments.test:
        
        logger.info(f"Prédictions")
        
        pred.prediction (seuil, predictions, matrice, modele, x_test, y)
        
        logger.info(f"FIN")
        
        