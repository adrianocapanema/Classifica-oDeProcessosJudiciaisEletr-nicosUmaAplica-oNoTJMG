# # Classificação automática de de Processos Judiciais Eletrônicos utilizando Redes Neurais (Deep Learning) - Glove

## Instalação de Pacotes
pip install elasticsearch_dsl
pip install pandasticsearch
pip install pandasticsearch[pandas]
pip install nltk
pip install joblib
pip install keras
conda install -c anaconda gensim
nltk.download('punkt')
nltk.download('stopwords')

## Importação de Pacotes
import time
tempo_inicial = time.time()
import pandas as pd
from pandasticsearch import DataFrame, Select
import numpy as np
from sklearn.model_selection import cross_val_score
import nltk
from joblib import dump, load
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split
import re
import os
import tensorflow as tf
print(tf.__version__)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, Activation, Dropout
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras import layers

## Carrega stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')

## carga base final pré-processada e balanceada
df = pd.read_csv("base_final.csv") 

## Pré-Processamento
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stopwords) # delete stopwors from text
    return text

df['conteudo_texto'] = df['conteudo_texto'].apply(clean_text)
plt.figure(figsize=(10,4))
plt.xlabel("Tipo de Documento")
plt.ylabel("Quantidade de Processos")
df.tipo.value_counts().sort_values(ascending=False).plot(kind='bar', title='Quantidade de Processos em cada tipo de Peça Processual');

## Início código para classificação de automática
def f_plota_perda_e_acuracia():

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
 
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#Obtendo precisão, F1, precisão e recall do modelo
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## Divisão do Dataset em Treino e Teste
processos_train, processos_test, train_Y, test_Y = train_test_split(df['conteudo_texto'],df['tipo'], test_size = 0.30, random_state = 42)
del(df)

# O número máximo de palavras a serem usadas. (mais frequentes)
max_words = 80000
oov_tok = '<OOV>'

# Número máximo de palavras em cada processo judicial
MAX_SEQUENCE_LENGTH = 1200  

# Cria o tokenizer
tokenize = text.Tokenizer(num_words=max_words, oov_token=oov_tok, char_level=False, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

## Criar um vetor por documento fornecido por entrada
tokenize.fit_on_texts(processos_train) 

##Imprime estatísticas da tokenização
print(tokenize.word_counts)
print(tokenize.document_count)
print(tokenize.word_index)
print(tokenize.word_docs)

word_index = tokenize.word_index
print('Encontrados %s tokens exclusivo.' % len(word_index))

## Transforma cada texto em uma sequência de números inteiros
X_train = tokenize.texts_to_sequences(processos_train)
X_test = tokenize.texts_to_sequences(processos_test)

##Trunca e preenche as seqüências de entrada para que todas tenham o mesmo comprimento na modelagem.
X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train.shape)

X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_test.shape)

##Converte rótulos categóricos em números.
Y_train = pd.get_dummies(train_Y)
print('Shape of label tensor:', Y_train.shape)
Y_test = pd.get_dummies(test_Y)
print('Shape of label tensor:', Y_test.shape)

## Define parâmetros da Rede Neural
embedding_dim = 100
vocab_size = len(tokenize.word_index) + 1  # Adding again 1 because of reserved 0 index
word_index = tokenize.word_index
batch_size = 150  #32 #64 # sinapses=150
epochs = 30       # 10 Desenvolvimento e 30 Produção # sinapses
drop = 0.5
num_filters = 512 #128

## Define o número de classes
num_classes = len(train_Y.unique())

## Cria e compila o modelo
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length=X_train.shape[1]))
model.add(layers.Conv1D(num_filters, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(Dropout(drop))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m,precision_m, recall_m])
print(model.summary())

##Treina a CNN
print('Treinando CNN...')
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=1, min_delta=0.0001)])


##Validação com os dados de teste
print('Validando...')
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing Loss:  {:.4f}".format(loss))
print("Testing f1_score:  {:.4f}".format(f1_score))
print("Testing precision:  {:.4f}".format(precision))
print("Testing recall:  {:.4f}".format(recall))

## Visualiza a Acurácia
f_plota_perda_e_acuracia()

tempo_final = time.time()
tempo_total = ((tempo_final - tempo_inicial) / 60)
print('-------------------------------------------------------------------------------------')
print("Tempo total de execução em minutos: %.2f" % tempo_total)
print('-------------------------------------------------------------------------------------')
