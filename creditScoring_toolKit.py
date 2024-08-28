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
def optimal_binning(base: pd.DataFrame, target: pd.DataFrame, variable, max_depth: int = 3, min_samples_leaf: int = 0.05):
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(base[[variable]].values.reshape(-1, 1), target)
    limites = tree.tree_.threshold[tree.tree_.threshold != -2]
    limites = np.sort(limites)
    limites = np.concatenate(([-np.inf], limites, [np.inf]))
    return limites


# Calcula los WOEs e IV para variables categoricas
def calculate_iv_cat(base: pd.DataFrame, target: pd.DataFrame, variable: str):
    # Agregamos target
    base['target'] = target.iloc[:, 0]
    grouped = base.groupby(variable, observed=False)[
        'target'].agg(['count', 'sum'])
    grouped['good'] = grouped['sum']
    grouped['bad'] = grouped['count'] - grouped['sum']
    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv'] = (grouped['good_dist'] -
                     grouped['bad_dist']) * grouped['woe']
    grouped['iv'] = np.where((grouped['bad'] == 0) | (
        grouped['good'] == 0), 0, grouped['iv'])
    grouped['iv_total'] = grouped['iv'].sum()
    grouped['dist'] = grouped['count'] / grouped['count'].sum()
    grouped['good_cat'] = grouped['good'] / grouped['count']
    grouped['bad_cat'] = grouped['bad'] / grouped['count']
    grouped['good_total'] = grouped['good'].sum() / grouped['count'].sum()
    iv = grouped['iv'].sum()

    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={variable: 'categoria'})

    grouped[['variable']] = variable
    grouped = grouped[['variable', 'categoria', 'count', 'sum', 'good', 'bad', 'good_dist',
                       'bad_dist', 'woe', 'iv', 'iv_total', 'dist', 'good_cat', 'bad_cat', 'good_total']]

    iv_df = pd.DataFrame({'variable': [variable], 'IV': [iv]})

    return iv_df, grouped


# Calcula los WOEs e IV para variables numericas
def calculate_iv_num(base: pd.DataFrame, target: pd.DataFrame, variable: str, bins):
    # Agregamos target
    base['target'] = target.iloc[:, 0]

    base['categoria'] = pd.cut(base[variable], bins=bins, duplicates='drop')
    grouped = base.groupby('categoria', observed=False)[
        'target'].agg(['count', 'sum'])
    grouped['good'] = grouped['sum']
    grouped['bad'] = grouped['count'] - grouped['sum']
    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv'] = (grouped['good_dist'] -
                     grouped['bad_dist']) * grouped['woe']
    grouped['iv'] = np.where((grouped['bad'] == 0) | (
        grouped['good'] == 0), 0, grouped['iv'])
    grouped['iv'] = np.where((grouped['bad'] == 0) | (
        grouped['good'] == 0), 0, grouped['iv'])
    grouped['iv_total'] = grouped['iv'].sum()
    grouped['dist'] = grouped['count'] / grouped['count'].sum()
    grouped['good_cat'] = grouped['good'] / grouped['count']
    grouped['bad_cat'] = grouped['bad'] / grouped['count']
    grouped['good_total'] = grouped['good'].sum() / grouped['count'].sum()
    iv = grouped['iv'].sum()

    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={variable: 'categoria'})

    grouped[['variable']] = variable
    grouped = grouped[['variable', 'categoria', 'count', 'sum', 'good', 'bad', 'good_dist',
                       'bad_dist', 'woe', 'iv', 'iv_total', 'dist', 'good_cat', 'bad_cat', 'good_total']]

    iv_df = pd.DataFrame({'variable': [variable], 'IV': [iv]})

    return iv_df, grouped
