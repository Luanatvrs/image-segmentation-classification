
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.feature_selection import f_classif
import pandas as pd
from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dados_colunas = ['class','region-centroid-col','region-centroid-row', 'region-pixel-count',
                  'short-line-density-5','short-line-density-2','vedge-mean', 'vedge-sd',
                  'hedge-mean','hedge-sd','intensity-mean','rawred-mean','rawblue-mean',
                'rawgreen-mean', 'exred-mean','exblue-mean','exgreen-mean',
                'value-mean', 'saturation-mean','hue-mean']

dados = pd.read_csv('segmentation.data',
                   skiprows=2,
                   header=None,
                   names=dados_colunas,
                   na_values='?',
                   lineterminator='\n')

# remove as linhas desnecessárias que veio com o nome das colunas no meio dos dados
dados = dados[dados['class'] != 'REGION-CENTROID-COL']
dados = dados[dados['class'] != 'REGION-CENTROID-COL ']
dados = dados[dados['class'].notnull()]

# reseta os índices depois de remover
dados.reset_index(drop=True, inplace=True)

# visualizar parte dos dados
print( dados.head() )

#verificando se os dados estão balanceados
print(dados['class'].value_counts(normalize=True) * 100)

Y_orig = dados['class'].copy()
X_orig = dados.iloc[:, 2:].copy()

X = dados.iloc[:, 2:]
Y = dados['class']

print(Y.unique())

print(X.head())

X_orig =  X.copy()
print(X_orig.head())

print(Y_orig.unique() )

#verificando se há dados ausentes
ausentes = (dados.drop(columns=['class']) == 0).sum()
print('\ndados ausentes:\n', ausentes)

#convertendo para NaN
cols = dados.columns[1:]
dados[cols] = dados[cols].replace(0, nan)
print(dados.isnull().sum())

#verificando se há dados com baixa representação
num_linhas = dados.shape[0]
for c in dados.columns:
  num_unicos = len( dados[c].unique() )
  percentage = float(num_unicos) / num_linhas * 100
  if percentage < 1:
    print('dados: %s,  %d, %.1f%%' % (c, num_unicos, percentage))


dados = dados.drop(columns=['region-pixel-count'])
dados = dados.drop(columns=['short-line-density-5'])
cols = dados.columns.drop('class')

# substituindo pela média
imputer = SimpleImputer(strategy='mean')
dados[cols] = imputer.fit_transform(dados[cols])

#verificando se há valores duplicados
dups = dados.duplicated()
print(dups[dups==True])
print('\ndados duplicados:\n')
print(dados[dups])

X = dados.iloc[:,2:]
Y = dados['class']

#Seleção de atributos por meio L1-based feature selection - LinearSVC
svc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000)
sfm = SelectFromModel(svc)
X_new = sfm.fit_transform(X, Y)

# normalização por meio do StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X) )

print(X_orig.head())
print(X.head())

from sklearn.model_selection import train_test_split
import numpy as np
print(Y_orig.value_counts())

# com os dados originais
X_oring_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(X_orig,
                      Y_orig, test_size=0.25, stratify=Y_orig,random_state=10)

# com os dados tratados
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    stratify=Y,random_state=10)

treinador = svm.SVC()  #algoritmo escolhido

modelo_orig = treinador.fit(X_oring_train, y_orig_train)

# predição com os mesmos dados usados para treinar
y_orig_pred = modelo_orig.predict(X_oring_train)
cm_orig_train = confusion_matrix(y_orig_train, y_orig_pred)
print('Matriz de confusão - com os dados ORIGINAIS usados no TREINAMENTO')
print(cm_orig_train)
print(classification_report(y_orig_train, y_orig_pred))

# predição com os mesmos dados usados para testar
print('Matriz de confusão - com os dados ORIGINAIS usados para TESTES')
y2_orig_pred = modelo_orig.predict(X_orig_test)
cm_orig_test = confusion_matrix(y_orig_test, y2_orig_pred)
print(cm_orig_test)
print(classification_report(y_orig_test, y2_orig_pred))

treinador = svm.SVC()  #algoritmo escolhido

modelo = treinador.fit(X_train, y_train)

# predição com os mesmos dados usados para treinar
y_pred = modelo.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred)
print('Matriz de confusão - com os dados TRATADOS usados no TREINAMENTO')
print(cm_train)
print(classification_report(y_train, y_pred))

# predição com os mesmos dados usados para testar
print('Matriz de confusão - com os dados TRATADOS usados para TESTES')
y2_pred = modelo.predict(X_test)
cm_test = confusion_matrix(y_test, y2_pred)
print(cm_test)
print(classification_report(y_test, y2_pred))