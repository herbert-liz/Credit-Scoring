# Creation Date: 24/08/2024
# Author: Herber Lizama
# Description: Support functions and methods for a logistic regression model in credit scoring

# Importing necessary libraries
import pandas as pd


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
