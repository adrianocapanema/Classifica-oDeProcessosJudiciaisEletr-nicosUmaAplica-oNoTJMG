## Instalação de Pacotes
import time
tempo_inicial = time.time()
import pandas as pd
from pandasticsearch import DataFrame, Select
from elasticsearch import Elasticsearch
from elasticsearch_dsl import connections, Search, Q, Boolean
from elasticsearch_dsl.query import MultiMatch, Match, Range, Bool
from sklearn.model_selection import cross_val_score, train_test_split
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import matplotlib.pyplot as plt
import re
import os

## Instancia stopwords e Stemmer
stopwords = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

diretorio_trabalho = "C:/Users/default/"
nome_arquivo_tipo_docto = 'tipo_documento.csv'
arquivo_tipo_docto = diretorio_trabalho + nome_arquivo_tipo_docto

## Busca dados no ElasticSearch
client = Elasticsearch(hosts='http://user:user@server:/')
q =  Q('bool',
         must=[
               Q('match', comarca='comarca'),
               ~Q('match', documentos__tipo='Petição'),
               Q('range', documentos__tamanho_binario_bytes={'gte': 2000})
             ]
        )
s = Search(using = client, index="index_name").query(q)
s = s.extra(track_total_hits=True)
response = s.execute()
print('Total de respostas encontradas %d.' % response.hits.total['value'])

## Pré-Processamento
results_df = pd.DataFrame((d.to_dict() for d in s.scan()))
df = pd.DataFrame()
for index, row in results_df.iterrows():
    for row_doc in row['documentos']:
        key = 'conteudo_texto'
        dict = row_doc
        if key in dict.keys():  
            if len(row_doc['conteudo_texto']) > 2000:
                df = df.append({'tipo' : row_doc['tipo'], 'conteudo_texto' : row_doc['conteudo_texto']}, ignore_index=True)
del (results_df)

## Estatísticas das Stop words
frases = df['conteudo_texto'].str.lower()
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
df_stops_words = pd.DataFrame()
cont = 0
for lista in textosQuebrados:
    for palavra in lista:
        if palavra in stopwords:
            new_row = {'palavra':palavra}
            ##append row to the dataframe
            df_stops_words = df_stops_words.append(new_row, ignore_index=True)
pd.set_option('display.max_rows', None)
df_stops_words.palavra.value_counts()
## Fim das Estatísticas das Stop words

df['conteudo_texto'] = df['conteudo_texto'].apply(clean_text)
filter_list = ['Petição Inicial', 'Sentença', 'Despacho', 'Decisão']
df = df[df.tipo.isin(filter_list)]
plt.figure(figsize=(10,4))
plt.xlabel("Tipo de Documento")
plt.ylabel("Quantidade de Processos")
df.tipo.value_counts().sort_values(ascending=False).plot(kind='bar', title='Quantidade de Processos em cada tipo de Peça Processual');
df_tipo_documento = pd.read_csv(arquivo_tipo_docto, encoding = 'Cp1252')
df['tipo'] = df['tipo'].map(df_tipo_documento.set_index('DS_TIPO_PROCESSO_DOCUMENTO')['ID_TIPO_PROCESSO_DOCUMENTO'])

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    ## Para não apresentar um resultado cada vez que roda
    SEED = 42        
    np.random.seed(SEED)

    tic = time.time()
    k = 10    
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)
    print('-------------------------------------------------------------------------------------')
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    print('-------------------------------------------------------------------------------------')
    tac = time.time()
    tempo_que_passou = ((tac - tic) / 60)
    print('-------------------------------------------------------------------------------------')
    print("Tempo de treinamento do modelo (" + nome + ") em minutos: %.2f" % tempo_que_passou)
    print('-------------------------------------------------------------------------------------')
        
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    tic = time.time()
    resultado = modelo.predict(validacao_dados)
    taxa_de_acerto = modelo.score(validacao_dados, validacao_marcacoes) * 100
    print('*************************************************************************')
    msg = 'Taxa de acerto do vencedor (' + str(modelo) + ') entre os dois algoritmos no mundo real: {0}'.format(taxa_de_acerto)
    print(msg)
    print('*************************************************************************')
    tac = time.time()
    tempo_que_passou = ((tac - tic) / 60)
    print('-------------------------------------------------------------------------------------')
    print("Tempo de predição do modelo no mundo real em minutos: %.2f" % tempo_que_passou)
    print('-------------------------------------------------------------------------------------')
    class_names = np.array(['Petição Inicial', 'Sentença', 'Despacho', 'Decisão'])
    print(classification_report(validacao_marcacoes, resultado, target_names=class_names))
    print('*************************************************************************')
    plot_confusion_matrix(validacao_marcacoes, resultado, class_names,
                      title='Matriz de Confusão')   

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
##Esta função imprime e plota a matriz de confusão.
##A normalização pode ser aplicada configurando `normalize = True`.
   if not title:
        if normalize:
            title = 'Matriz de confusão normalizada'
        else:
            title = 'Matriz de confusão sem normalização'

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão Normalizada")
    else:
        print('Matriz de confusão sem normalização')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Mostrando todos os rótulos
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Giro dos rótulos de escala e seu alinhamento.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # dimensões dos dados e anotações de texto.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

## Cria a transformação
vectorizer = TfidfVectorizer()

## tokenize e constroi o vocabulário
vectorizer.fit(processos_train)

## Tamanho dicionário
len(vectorizer.vocabulary_.keys())
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

## Transforma cada texto em uma sequência de números inteiros
treino_dados = vectorizer.transform(processos_train)
treino_marcacoes  = Y_train
validacao_dados = vectorizer.transform(processos_test)
validacao_marcacoes  = Y_test

## Classificadores
resultados = {}
v_max_iter = 8000

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=42, max_iter=v_max_iter))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=42, max_iter=v_max_iter))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier(random_state=42)
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print(resultados)
maximo = max(resultados)
vencedor = resultados[maximo]
print('**************')
print("Vencedor: ")
print(vencedor)
print('**************')

## Treinando o modelo final (vencedor)
vencedor.fit(treino_dados, treino_marcacoes)

## Salva modelo Vencedor
dump(vencedor, arquivo_modelo_salvo) 
np.set_printoptions(precision=2)

tempo_final = time.time()
tempo_total = ((tempo_final - tempo_inicial) / 60)
print('-------------------------------------------------------------------------------------')
print("Tempo total de execução em minutos: %.2f" % tempo_total)
print('-------------------------------------------------------------------------------------')

## Teste real com dados de validação
teste_real(vencedor, validacao_dados, validacao_marcacoes)

modeloOneVsRest.fit(treino_dados, treino_marcacoes)
teste_real(modeloOneVsRest, validacao_dados, validacao_marcacoes)

modeloMultinomial.fit(treino_dados, treino_marcacoes)
teste_real(modeloMultinomial, validacao_dados, validacao_marcacoes)

modeloAdaBoost.fit(treino_dados, treino_marcacoes)
teste_real(modeloAdaBoost, validacao_dados, validacao_marcacoes)