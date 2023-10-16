# Test_DS

[Lien GitLab](https://gitlab.com/SerenaNGUEDIA/test_ds)


## Description du programme

J'ai mis en place un programme qui permet d'entrainer un modèle et fournir des prédictions.

Commandes respectives pour installer les librairies necessaires et lancer le modèle:

```
pip install -r requirement.txt

python code/run.py --config conf/config.yml --train --test

```


- [ ] **requirement.txt :** est un fichier qui contient les librairies à installer pour pouvoir lancer le programme.
- [ ] **notebook.ipynb :** est le notebook d’exploration afin de partager ma démarche : exploration des données, analyses des variables, choix du model et ses performances, matrice de confusion...
- [ ] **data :** est le repertoire qui contient les données: *data.csv* (données initiales), *train.csv* (données de l'entraînement), *test.csv* (données pour les prédictions).
- [ ] **output :** est le repertoire dans lequel s'enregistrent: le modèle entrainé (*modele.pickle*), les logs du programme (*file.log*), les prédictions (*predictions.csv*) et la matrice de confusion (*confusion_matrix.csv*).
- [ ] **conf/config.yml :** est le fichier de configuration qui renferme les variables modifiables.
- [ ] **code/training_class.py :** est la classe pour entrainer le modèle.
- [ ] **code/test_class.py :** est la classe qui utilise le modèle entrainé pour fournir des prédictions.
- [ ] **code/pretraitement.py :** permet de lire les variables du fichier de configuration, lire les données d'apprentissage et de test (si elles existent) ou les créer à partir du dataframe initial, déterminer les meilleurs paramètres du modèle.
- [ ] **code/fonctions.py :** renferme les fonctions nécessaires.
- [ ] **code/apprentissage.py :** permet de faire l'apprentissage à partir des données *train.csv*
- [ ] **code/prediction.py :** permet de faire les prédictions à partir des données *test.csv*
- [ ] **code/run.py :** est le programme d'exécution.




## Réponse à la question posée

```
Comment auriez-vous modélisé la problématique si vous aviez eu en plus une variable 
de plus indiquant l'année et le mois (exemple : identifiant_mois = 202101) 
et donc une base représentant plusieurs vues mensuelles pour le même client id ?

```

- [ ] Si on a une base représentant plusieurs vues mensuelles des clients, alors cela voudrait dire que nous avons des données qui évoluent avec le temps et donc qui représentent des séries temporelles. Dans ce cas, un modèle classique de Machine Learning ne serait pas adapté et il faudrait plutôt passer par un modèle de réseaux de neurones. Pour ma part, j'aurais opté pour un RNN (réseau de neurones récurrents) et plus précisement un modèle de LSTM (Long Short-Term Moemory). Les LSTM sont connus pour surmonter la dépendance à long terme et supprimer les inconvénients des RNN. De plus, ils auraient été plus efficace pour traiter les variables catégorielles.

