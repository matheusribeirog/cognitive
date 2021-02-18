# pip install pandas, numpy, ipython, scikit-learn, matplotlib

#Sobre Spyder
# ctrl + P
# F5 / F9

import pandas as pd
import numpy as np
from IPython.display import Image, display

# Funcao
# https://www.w3schools.com/python/python_functions.asp
def soma_funcao(x, y):
    return x + y
resultado_soma_funcao = soma_funcao(1, 2)
print('resultado_soma_funcao: ', resultado_soma_funcao)

# Lambda
# https://www.w3schools.com/python/python_lambda.asp
soma_lambda  = lambda x, y: x + y
resultado_soma_lambda = soma_lambda(3, 4)
print('resultado_soma_lambda: ',resultado_soma_lambda)

# Iteracao
for item in [1, 2, 3]:
    print('iteracao: ', item)

# Listas Python
lista1 = [1, 2 ,3]
resultado1 = [item+1 for item in lista1]
print('resultado1: ', resultado1)
resultado2 = list(map((lambda item: item + 1), lista1))
print('resultado2: ', resultado2)

display(Image('numpy_array_pandas_dataframe.jpeg', width=600, height=600))
# Numpy
# https://www.w3schools.com/python/numpy_intro.asp
ndarray_v = np.array([4, 5, 6])
print('numpy adrray_v: ', ndarray_v)
print(type(ndarray_v))
print(ndarray_v.__class__)

ndarray_m = np.array([[4, 5, 6], [7, 8, 9]])
print('numpy adrray_m: ', ndarray_m)

# https://www.w3schools.com/python/pandas/default.asp
# https://www.w3schools.com/python/pandas/pandas_series.asp
numeros = [10, 11, 12]
serie_n = pd.Series(numeros, index = ["x", "y", "z"])
print('serie_n: ', serie_n)
print('serie_n["x"]: ', serie_n["x"])

dic_numeros = {"x": 13, "y":14, "z":15}
serie_d = pd.Series(dic_numeros)
print('serie_d["x"]: ', serie_d["x"])

# Series entao resumidamente sao colunas
dic_dados = {"a": [1, 2], "b": [3, 4], "c": [5, 6] }
df_d = pd.DataFrame(dic_dados)
print('df_d: ', df_d)

# Dados coluna
print('type(df_d["a"]): ', type(df_d["a"]))
print('df_d["a"]: ', df_d["a"])

# Aplicando transformacao
multiplica_2 = lambda x: x*2
df_d["nova_coluna_b"] = df_d["b"].apply(multiplica_2)
print('df_d["nova_coluna_b"]: ', df_d["nova_coluna_b"])

# Leitura de arquivo
dataset = pd.read_csv('Case_cobranca.csv')