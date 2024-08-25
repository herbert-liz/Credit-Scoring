# Creation Date: 24/08/2024
# Author: Herber Lizama
# Description: Support functions and methods for a logistic regression model in credit scoring

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree


# Crea una lista con las variables categoricas y numericas
def var_numericas_categoricas(base: pd.DataFrame, drop_col: list = []):
    numericas = []
    categoricas = []

    for variable in base.drop(columns=drop_col).columns:
        if base[variable].dtype in ('float64', 'int64'):
            numericas.append(variable)
        else:
            categoricas.append(variable)

    return numericas, categoricas


# Calcula las categorias optimas para variables numerica usando arboles de decision
def optimal_binning(base: pd.DataFrame, variable, target, max_depth: int = 3, min_samples_leaf: int = 0.05):
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(base[[variable]].values.reshape(-1, 1), base[[target]])
    limites = tree.tree_.threshold[tree.tree_.threshold != -2]
    limites = np.sort(limites)
    limites = np.concatenate(([-np.inf], limites, [np.inf]))
    return limites
