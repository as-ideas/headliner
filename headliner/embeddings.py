from typing import Dict

import numpy as np


def read_embedding(file_path: str, vector_dim: int) -> Dict[str, np.array]:
    """
    Reads an embedding file in glove format into a dictionary mapping tokens to vectors.
    """

    glove = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            # handle weird whitespaces in tokens
            if len(values[1:]) > vector_dim:
                wordcount = len(values) - vector_dim
                vec = np.asarray(values[wordcount:], dtype='float32')
            else:
                vec = np.asarray(values[1:], dtype='float32')
            token = values[0]
            glove[token] = vec
    return glove


def embedding_to_matrix(embedding: Dict[str, np.array],
                        token_index: Dict[str, int],
                        embedding_dim: int) -> np.array:
    """
    Converts an embedding dictionary into a weights matrix used to initialize an embedding layer.
    It ensures that all tokens in the token_index dictionare are mapped to a row, even those that are
    not contained in the provided embedding dictionary. Unknown tokens are initialized with a random
    vector with entries between -1 and 1.

    Args:
        embedding: dictionary mapping tokens to embedding vectors
        token_index: dictionary mapping tokens to indices that are fed into the embedding layer
        embedding_dim: size of the embedding vectors

    Returns: embedding weights as numpy array
    """
    np.random.seed(42)
    embedding_matrix = 2. * np.random.rand(len(token_index) + 1, embedding_dim) - 1.
    for token, index in token_index.items():
        embedding_vec = embedding.get(token)
        if embedding_vec is not None:
            embedding_matrix[index] = embedding_vec
    return embedding_matrix
