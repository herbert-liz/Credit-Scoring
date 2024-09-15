# Creation Date: 24/08/2024
# Author: Herber Lizama
# Description: Support functions and methods for a logistic regression model in credit scoring

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree


# Lista de variables categoricas
def var_numericas_categoricas(base: pd.DataFrame, drop_col: list = []):
    numericas = []
    categoricas = []

    for variable in base.drop(columns=drop_col).columns:
        if base[variable].dtype in ('float64', 'int64'):
            numericas.append(variable)
        else:
            categoricas.append(variable)

    return numericas, categoricas

# Calcula los WOEs e IV para variables categoricas


def calculate_iv_cat(base: pd.DataFrame, variable: str, target: str):
    grouped = base.groupby(variable, observed=False)[target].agg(cantidad='count', good='sum')
    grouped['bad'] = grouped['cantidad'] - grouped['good']
    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    grouped['iv'] = np.where((grouped['bad'] == 0) | (grouped['good'] == 0), 0, grouped['iv'])
    grouped['iv_total'] = grouped['iv'].sum()
    grouped['dist'] = grouped['cantidad'] / grouped['cantidad'].sum()
    grouped['good_cat'] = grouped['good'] / grouped['cantidad']
    grouped['bad_cat'] = grouped['bad'] / grouped['cantidad']
    grouped['good_total'] = grouped['good'].sum() / grouped['cantidad'].sum()
    iv = grouped['iv'].sum()

    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={variable: 'categoria'})

    grouped[['variable']] = variable
    grouped = grouped[['variable', 'categoria', 'cantidad', 'good', 'bad', 'good_dist',
                       'bad_dist', 'woe', 'iv', 'iv_total', 'dist', 'good_cat', 'bad_cat', 'good_total']]

    iv_df = pd.DataFrame({'variable': [variable], 'IV': [iv]})

    return iv_df, grouped


# Calcula las categorias optimas para variables numerica usando arboles de decision
def optimal_binning(base: pd.DataFrame, target: pd.DataFrame, variable, max_depth: int = 3, min_samples_leaf: int = 0.05):
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(base[[variable]].values.reshape(-1, 1), target)
    limites = tree.tree_.threshold[tree.tree_.threshold != -2]
    limites = np.sort(limites)
    limites = np.concatenate(([-np.inf], limites, [np.inf]))
    return limites

# Calcula los WOEs e IV para variables numericas


def calculate_iv_num(base: pd.DataFrame, target: pd.DataFrame, variable: str, bins):
    # Agregamos target
    base['target'] = target.iloc[:, 0]

    base['categoria'] = pd.cut(base[variable], bins=bins, duplicates='drop')
    grouped = base.groupby('categoria', observed=False)[
        'target'].agg(['cantidad', 'sum'])
    grouped['good'] = grouped['sum']
    grouped['bad'] = grouped['cantidad'] - grouped['sum']
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
    grouped['dist'] = grouped['cantidad'] / grouped['cantidad'].sum()
    grouped['good_cat'] = grouped['good'] / grouped['cantidad']
    grouped['bad_cat'] = grouped['bad'] / grouped['cantidad']
    grouped['good_total'] = grouped['good'].sum() / grouped['cantidad'].sum()
    iv = grouped['iv'].sum()

    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={variable: 'categoria'})

    grouped[['variable']] = variable
    grouped = grouped[['variable', 'categoria', 'cantidad', 'sum', 'good', 'bad', 'good_dist',
                       'bad_dist', 'woe', 'iv', 'iv_total', 'dist', 'good_cat', 'bad_cat', 'good_total']]

    iv_df = pd.DataFrame({'variable': [variable], 'IV': [iv]})

    return iv_df, grouped


# Crea base WOEs
def base_woes(clave: str, variable: str, base_variables: pd.DataFrame, base_categorias: pd.DataFrame):
    # Para variables categoricas
    if base_variables[variable].dtype in ('float64', 'int64'):
        # Filtrar las categorías para la variable especificada
        categorias = base_categorias[base_categorias['variable'] == variable]
        categorias = categorias[['categoria', 'woe']]

        # Crear los intervalos a partir de la columna 'bin'
        bin_intervals = pd.IntervalIndex.from_tuples(
            [parse_bin(b) for b in categorias['categoria']])

        # Crear el mapeo de intervalos a valores WOE
        woe_mapping = pd.Series(categorias['woe'].values, index=bin_intervals)

        # Crear una copia del DataFrame base_variables para agregar la columna WOE
        categoria_woe = base_variables[[clave, variable]].copy()

        # Asignar los valores WOE a la variable especificada en el DataFrame base_variables
        categoria_woe[f'{variable}_woe'] = pd.cut(
            categoria_woe[variable], bins=bin_intervals).map(woe_mapping)

    # Para variables numericas
    else:
        # Filtrar las categorías para la variable especificada
        categorias = base_categorias[base_categorias['variable'] == variable]
        categorias = categorias[['categoria', 'woe']]

        # Crear el mapeo de intervalos a valores WOE
        woe_mapping = pd.Series(
            categorias['woe'].values, index=categorias['categoria'])

        # Crear una copia del DataFrame base_variables para agregar la columna WOE
        categoria_woe = base_variables[[clave, variable]].copy()

        # Asignar los valores WOE a la variable especificada en el DataFrame base_variables
        categoria_woe[f'{variable}_woe'] = categoria_woe[variable].map(
            woe_mapping)

    return categoria_woe[[clave, f'{variable}_woe']]
