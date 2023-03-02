'''
- Implement the query likelihood language model with
    (a) Laplace smoothing,
    (b) Lidstone correction with ϵ = 0.1
    (c) Dirichlet smoothing with µ = 50

- Retrieve 100 passages from passages for each test query.
- Store outcomes in the files laplace.csv, lidstone.csv, and dirichlet.csv.
    column score should report the natural logarithm of the probability scores
'''
import numpy as np
from scipy.sparse import csr_matrix

from task3 import Help, select_first100, passages_indexes, queries_indexes


def get_smoothed_tf(inverted_indexes, smooth_type: str, epsilon=None, np_log=np.log):
    vocab_n, doc_n = inverted_indexes.shape
    epsilon = 0. if smooth_type == 'dirichlet' else 1. if smooth_type == 'laplace' else (epsilon or .1)

    temp = inverted_indexes.copy()
    temp.data += epsilon

    # (vocab_n, doc_n)
    result = Help.scale_csr(temp, 1 / (doc_len + epsilon * vocab_n))

    result.data = np_log(result.data)

    return result


class DirichletScore:
    def __init__(self, prob_passages: csr_matrix, mu=50):
        # (1, doc_n)
        denominator = doc_len + mu
        self.coef0 = doc_len / denominator
        self.coef1 = mu / denominator

        # (vocab_n, doc_n)
        self.prob_passages = prob_passages
        self.temp0 = Help.scale_csr(prob_passages, self.coef0)  # (vocab_n, doc_n)

        self.freq_of_vocab = passages_indexes.sum(axis=1)  # (vocab_n, 1)
        self.n_tokens = self.freq_of_vocab.sum()

    def __getitem__(self, index):
        passages_i = index[1] if isinstance(index, tuple) else index

        # (vocab_n, 1)
        coef1 = self.coef1[:, passages_i]
        temp1 = self.freq_of_vocab @ coef1  # (vocab_n, 1) @ (1, doc_n) --> (vocab_n, doc_n)
        temp1.data /= self.n_tokens
        result = np.array((self.temp0[:, passages_i] + temp1).sum(axis=0)).reshape(-1)

        return result


def get_scores(prob_passages: csr_matrix):
    return (queries_indexes.T @ prob_passages).toarray()


doc_len = passages_indexes.sum(axis=0)  # (1, doc_n)

if __name__ == '__main__':
    print('Calculate query likelihood')

    types = ['laplace','lidstone','dirichlet']

    for t in types:
        print(f'- {t}')
        prob_passages = get_smoothed_tf(passages_indexes, t)

        score = DirichletScore(prob_passages) if t == 'dirichlet' else get_scores(prob_passages)

        result = select_first100(score, remove_negative=False)

        # Store outcomes in the files laplace.csv, lidstone.csv, and dirichlet.csv.
        result.to_csv(f'{t}.csv', header=False, index=False)

