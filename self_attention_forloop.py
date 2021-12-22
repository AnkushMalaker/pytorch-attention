import numpy as np
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """Takes embeddings of tokens as input and outputs more contextualized embeddings.
    This module can be stacked and has no weights, but is expensive for longer sequence lengths due to being O(n^2)"""

    def __init__(self, ) -> None:
        super().__init__()

    def self_attention(self, query, keys) -> torch.Tensor:
        weights = torch.stack([torch.dot(query, keys[i]) for i in range(len(keys))])
        weights = weights / torch.sum(weights)
        return weights

    def forward(self, sentence):
        # Say sentence is a Tensor of shape (None, 3, 4) for (batch_size, number_of_words, dim)
        # We have to loop over all vectors, here, 3 and calculate dot product with the 3 vectors.
        # This is an n^2 operation.
        weights = torch.stack([self.self_attention(query, sentence) for query in sentence])
        reweighted_vectors = torch.matmul(weights, sentence)
        return reweighted_vectors

if __name__ == "__main__":
    word_1 = np.array([1, 0, 0])
    word_2 = np.array([0, 1, 0])
    word_3 = np.array([1, 1, 0])
    word_4 = np.array([0, 0, 1])

    sentence = [word_1, word_2, word_3, word_4]

    print("Initial vectors: ")
    for word in sentence:
        print(word)
    print("Reweighted vectors: ")
    sa = SelfAttention()
    sentence = torch.Tensor(np.array(sentence)) # Convert words to tensor to feed to pytorch module
    print(sa(sentence))