# # Classificação automática de de Processos Judiciais Eletrônicos utilizando Redes Neurais (Deep Learning) - CNN/Glove


## Importa pacotes
from gensim.models import KeyedVectors

## carga vetor de palavras Glove
model_glove = KeyedVectors.load_word2vec_format(r"C:\Users\dafault\\glove_s100.txt")

## Exibe o tamanho do vetor de palavras
model_glove.vector_size

## Define o tamanho do vocabulário
embedding_dim = 100
vocab_size = len(tokenize.word_index) + 1  # Adding again 1 because of reserved 0 index
word_index = tokenize.word_index

## Preenche o vacabulário com zeros
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

## Define o número de classes
num_classes = len(train_Y.unique())

## Cria e compila o modelo
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,
                    weights=[embedding_matrix],
                    input_length=X_train.shape[1],
                    trainable=True))
model.add(layers.Conv1D(num_filters, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(Dropout(drop))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m,precision_m, recall_m])
print(model.summary())

## Treina a CNN
print('Treinando CNN...')
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=1, min_delta=0.0001)])

## Validação com os dados de teste
print('Validando...')
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print("Teste Acurácia:  {:.4f}".format(accuracy))
print("Teste Perda:  {:.4f}".format(loss))
print("Teste Score F1:  {:.4f}".format(f1_score))
print("Teste precisão:  {:.4f}".format(precision))
print("Teste recall:  {:.4f}".format(recall))

## Visualiza a Acurácia
f_plota_perda_e_acuracia()
