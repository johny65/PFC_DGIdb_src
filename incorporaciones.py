

def incorporacion_glove():
    '''
    Carga los vectores de embedding preentrenados de GloVe
    '''
    
    embeddings_index = dict()
    f = open('glove.6B.300d.txt',encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((TOP_PALABRAS_FRECUENTES,DIMENSION_VECTORES_EMBEDDING))
    for word, index in tokenizer.word_index.items():
        if index > TOP_PALABRAS_FRECUENTES - 1:
            break
        else:   
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector  


def sample_to_number():

    return 