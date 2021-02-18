# pip install pandas, numpy, ipython, scikit-learn, matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix

dataset = pd.read_csv('Case_cobranca.csv')

colunas = list(dataset.columns)

# PRE-PROCESSAMENTO
# Criando a feature alvo
# Os títulos para serem considerados bons devem ter sido pagos em até 90 dias de atraso.
menor_noventa_dias_atraso = lambda x: 0 if(np.isnan(x) or x > 90) else 1
dataset['ALVO'] = dataset['TEMP_RECUPERACAO'].apply(menor_noventa_dias_atraso)


display(Image('tipos_variaveis.jpg', width=600, height=600))


# FEATURE ENG - Quantitativa
minimo_18anos = lambda x: 18 if(np.isnan(x) or x < 18) else x
dataset['PRE_IDADE'] = dataset['IDADE'].apply(minimo_18anos)

display(Image('percentil.png', width=600, height=600))
resumo_dataset = dataset.describe(percentiles=[.99])
normaliza_idade = lambda x: 1. if(x > 76) else (x-18)/(76-18)
dataset['PRE_IDADE'] = dataset['PRE_IDADE'].apply(normaliza_idade)

resumo_dataset = dataset.describe(percentiles=[.99])
qtde_dividas_normalizado = lambda x: 0 if(np.isnan(x)) else x / 16
dataset['PRE_QTDE_DIVIDAS'] = dataset['QTD_DIVIDAS'].apply(qtde_dividas_normalizado)


# FEATURE ENG - Qualitativa (forma nao otimizada)
display(Image('escalar_vetorizado.png', width=600, height=600))
display(Image('categorico_dummy.png', width=600, height=600))
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]    
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]                        
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]

# Divisao dataset: Treinamento / Teste
display(Image('treino_teste_divisao.png', width=600, height=600))
display(Image('generalizacao.png', width=600, height=600))

y = dataset['ALVO']              # Carrega alvo ou dataset.iloc[:,7].values
X = dataset.iloc[:, 8:15].values # Carrega colunas 8, 9, 10, 11, 12, 13 e 14 (a 15 não existe até este momento)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 25)


# Treino
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree.fit(X_train, y_train)

# Teste
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)


plot_confusion_matrix(dtree, X_test, y_test)
print(classification_report(y_test, y_pred_test_DT))