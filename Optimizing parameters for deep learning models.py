## Instala Bibliotecas
pip install elasticsearch_dsl
pip install pandasticsearch
pip install pandasticsearch[pandas]
pip install nltk
pip install joblib
pip install keras
conda install -c anaconda gensim
nltk.download('punkt')
nltk.download('stopwords')
conda install gensim==2.0.0
conda install scikit-learn=0.22

## Importa Bibliotecas
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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from gensim.models import KeyedVectors

## Instancia stop words em Português
stopwords = nltk.corpus.stopwords.words('portuguese')

## Carga base final balanceada em Excel
df = pd.read_csv("base_final.csv") 

## Realiza o Pré-Processamento
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

## Início código para classificação de automática
## Função com modelo CNN/GLOVE

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dropout(drop))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

## Divisão do Dataset em Treino e Teste
processos_train, processos_test, train_Y, test_Y = train_test_split(df['conteudo_texto'],df['tipo'], test_size = 0.30, random_state = 42)

print('processos_train shape:', processos_train.shape)
print('processos_test shape:', processos_test.shape)
print('train_Y shape:', train_Y.shape)
print('test_Y shape:', test_Y.shape)

del(df)

## O número máximo de palavras a serem usadas. (mais frequentes)
max_words = 80000
oov_tok = '<OOV>'

## Número máximo de palavras em cada processo.
MAX_SEQUENCE_LENGTH = 1200

## Cria o tokenizer
tokenize = text.Tokenizer(num_words=max_words, oov_token=oov_tok, char_level=False, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

## Documentos de codificação inteira
## Cria um vetor por documento fornecido por entrada
tokenize.fit_on_texts(processos_train) # Treina somente dados de teste

## Imprime estatísticas do Tokenizer
print(tokenize.word_counts)
print(tokenize.document_count)
print(tokenize.word_index)
print(tokenize.word_docs)

word_index = tokenize.word_index
print('Encontrados %s tokens exclusivos' % len(word_index))

## Transforma cada texto em uma sequência de números inteiros
X_train = tokenize.texts_to_sequences(processos_train)
X_test = tokenize.texts_to_sequences(processos_test)

## Trunca e preenche as seqüências de entrada para que todas tenham o mesmo comprimento para modelagem.
X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train.shape)

X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_test.shape)

#Converte rótulos categóricos em números.
Y_train = pd.get_dummies(train_Y)
print('Shape of label tensor:', Y_train.shape)

Y_test = pd.get_dummies(test_Y)
print('Shape of label tensor:', Y_test.shape)

## Importa vetores de palavras Glove no idioma Português
model_glove = KeyedVectors.load_word2vec_format(r"C:\Users\default\glove_s100.txt")

## Define Vocabulário
embedding_dim = 100
vocab_size = len(tokenize.word_index) + 1  
word_index = tokenize.word_index

## Preenche incorparação de palavras com zeros
embedding_matrix = np.zeros((vocab_size, embedding_dim))

##Considera apenas vetores palavras GLOVE que que também estão na base de dados judicial
for pair in zip(model_glove.index2word, model_glove.syn0):
    word = pair[0]
    vector = pair[1]
    if word in word_index:
        idx = word_index[word] 
        embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

print('Incorporação de palavras não nulas: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print('Cobertura do vocabulário pelo modelo pré-treinado: ', (nonzero_elements / vocab_size) * 100)

## Define parâmetros da Rede Neural
batch_size = 150 
epochs = 30 
drop = 0.5
num_filters = 512 
maxlen = X_train.shape[1]
output_file = 'data/output.txt'

## Grade de parâmetros para pesquisa em grade
param_grid = dict(num_filters=[32, 64, 128, 512],
                  kernel_size=[3, 5, 7],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  maxlen=[maxlen])

## Define o número de classes
num_classes = len(train_Y.unique())


## Instância o modelo
model = KerasClassifier(build_fn=create_model,
                        epochs=epochs, batch_size=batch_size,
                        verbose=False)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                          cv=10, verbose=1, n_iter=5)

## Treina cada modelo
grid_result = grid.fit(X_train, Y_train)

tempo_final = time.time()
tempo_total = ((tempo_final - tempo_inicial) / 60)
print('-------------------------------------------------------------------------------------')
print("Tempo total de execução em minutos: %.2f" % tempo_total)
print('-------------------------------------------------------------------------------------')

## Avalia conjunto de testes
test_accuracy = grid.score(X_test, Y_test)

## Salva e avalia os resultados
with open(output_file, 'a') as f:
    s = ('Executando {} Melhor Acurácia : '
         '{:.4f}\n{}\nTeste Acurácia : {:.4f}\n\n')
    output_string = s.format(
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy)
    print(output_string)
    f.write(output_string)

tempo_final = time.time()
tempo_total = ((tempo_final - tempo_inicial) / 60)
print('-------------------------------------------------------------------------------------')
print("Tempo total de execução em minutos: %.2f" % tempo_total)
print('-------------------------------------------------------------------------------------')