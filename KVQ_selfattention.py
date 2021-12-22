import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax

class KVQ_self_attention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.WQ = Parameter(torch.empty(embedding_dim, embedding_dim))
        self.WK = Parameter(torch.empty(embedding_dim, embedding_dim))
        self.WV = Parameter(torch.empty(embedding_dim, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.WQ, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.WK, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.WV, a=math.sqrt(5))

    # def kvq_selfattention(query, sentence):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # suppose input is 4 words, ie, (batch, 4, 5) tensor if dimention size if 5
        # WQ, WK, WV are 5,5 matricies. We can matmul(input, WQ) to get query vectors for all inputs
        # And similar for keys and values
        query = torch.matmul(input, self.WQ)  # 4, 5
        keys = torch.matmul(input, self.WK)  # 4, 5
        values = torch.matmul(input, self.WV)  # 4, 5

        # Calculate attention scores now
        attn_scores = torch.matmul(query, keys.T)  # 4, 4
        attn_scores_softmax = softmax(attn_scores, dim=-1)  # 4, 4

        # I dont understand this next part still, tbh but I'm tired now.
        weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]

        return weighted_values.sum(dim=0)


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
    sa = KVQ_self_attention(embedding_dim=3)
    sentence = torch.Tensor(
        np.array(sentence)
    )  # Convert words to tensor to feed to pytorch module
    print(sa(sentence))
